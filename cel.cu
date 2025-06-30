#include "zksoftmax.cuh"
#include "zkfc.cuh"
#include "fr-tensor.cuh"
#include "proof.cuh"
#include "commitment.cuh"
#include "rescaling.cuh"
#include <string>


int main(int argc, char *argv[])
{    
 //   cudaSetDevice(3);
    // 从命令行参数获取 vocab_size 和 seq_len
    uint vocab_size = std::stoi(argv[1]);
    string one_hot_file = argv[2];
    string input_file_name = argv[3];
    uint seq_len = std::stoi(argv[4]);
    uint embed_dim = std::stoi(argv[5]);
    string workdir = argv[6];
    string output_file_name = argv[7];

    auto lm_weight = create_weight(
        workdir + "/" + "lm_head.weight-pp.bin",
        workdir + "/" + "lm_head-weight-int.bin",
        workdir + "/" + "lm_head-weight-commitment.bin",
        seq_len, vocab_size
    );
    /*logits*/

    zkFC output_layer(4096, vocab_size, lm_weight.weight);

    FrTensor input = FrTensor::from_int_bin(input_file_name);
    Rescaling output_rescale(1 << 16);
    auto logits = output_layer(input);
    auto logits_ = output_rescale(logits);
    output_rescale.prove(logits, logits_);

    cout << "logits proof successfully verified!" << endl; 
    
    /*softmax*/
    auto padded_logits = pad_to_power_of_two(logits, seq_len, vocab_size);
    //cout << "padded_logits" <<padded_logits.size<< endl; 

    zkSoftmax softmax({1<<8, 1<<20, 1<<20}, 1, 0, 1UL<<32, {1<<18, 1<<22}, pow(2, ceil(log2(vocab_size))), seq_len, pow(2, ceil(log2(vocab_size))), 1);
    //zkSoftmax softmax({1<<8, 1<<8, 1<<20}, 1, 0, 1UL<<32, {1<<16, 1<<20}, vocab_size, seq_len, pow(2, ceil(log2(vocab_size))), 1);

    FrTensor shift(pow(2, ceil(log2(vocab_size)))), logits_shifted(seq_len * pow(2, ceil(log2(vocab_size))));    
   // FrTensor shift(vocab_size), logits_shifted(seq_len * vocab_size);    
    
    vector<FrTensor> logits_segments, Y_segments, m_segments;
    FrTensor Y = softmax.compute(padded_logits, shift, logits_shifted, logits_segments, Y_segments, m_segments);

    auto temp_rand = random_vec(3);
    vector<Polynomial> proof;
    cout << "Y" <<Y.size<< endl; 

    /*softmax.prove(Y, logits, shift, logits_shifted, logits_segments, Y_segments, m_segments, 
    random_vec(26), random_vec(26), temp_rand[0], temp_rand[1], temp_rand[2], proof);*/
    softmax.prove(Y, padded_logits, shift, logits_shifted, logits_segments, Y_segments, m_segments, 
    random_vec(26), random_vec(26), temp_rand[0], temp_rand[1], temp_rand[2], proof);
    cout << "softmax proof successfully verified!" << endl; 

 /*1.	交叉熵损失对outputnorm的梯度*/ 


    FrTensor one_hot = FrTensor::from_int_bin("one_hot_matrix.bin");

    auto one_hot_logits = pad_to_power_of_two(one_hot, seq_len, vocab_size);
    auto lm_weight_weight_padded = pad_to_power_of_two(lm_weight.weight, vocab_size, embed_dim);

    auto GD_logistin=Y-one_hot_logits;
    Y.~FrTensor();
    one_hot_logits.~FrTensor();

    //auto GD_outputnorm_out = FrTensor::matmul(Y, GD_logistin.transpose(seq_len, pow(2, ceil(log2(vocab_size)))), seq_len, pow(2, ceil(log2(vocab_size))), seq_len);
    auto GD_outputnorm_out = FrTensor::matmul( GD_logistin,lm_weight_weight_padded.transpose(embed_dim,pow(2, ceil(log2(vocab_size)))), seq_len,pow(2, ceil(log2(vocab_size))),  embed_dim);

    auto u1 = random_vec(ceilLog2(seq_len)); /*随机向量*/
    auto u2 = random_vec(ceilLog2(embed_dim)); /*随机向量*/
    auto ud = random_vec(ceilLog2(pow(2, ceil(log2(vocab_size)))));

    auto claim = GD_outputnorm_out.multi_dim_me({u1, u2}, {seq_len, embed_dim});/*MLE*/

    //auto final_claim = zkip(claim, Y.partial_me(u1,seq_len, pow(2, ceil(log2(vocab_size)))), GD_logistin.transpose(seq_len, pow(2, ceil(log2(vocab_size)))).partial_me(u2, seq_len, 1), ud, proof);  
    auto final_claim = zkip(claim,GD_logistin.partial_me(u1, seq_len, pow(2, ceil(log2(vocab_size)))), lm_weight_weight_padded.transpose(embed_dim,pow(2, ceil(log2(vocab_size)))).partial_me(u2,embed_dim, 1),  ud, proof);  

/*	损失对MLP输出 H 的梯度*/
   auto rmsnorm_weight = create_weight(
        workdir + "/" + "model.norm.weight-pp.bin",
        workdir + "/" + "model-norm-weight-int.bin",
        workdir + "/" + "model-norm-weight-commitment.bin",
        1, embed_dim
    );    
    FrTensor rms_inv_temp = FrTensor::from_int_bin("GD_rms_inv_temp_for_output.bin");


    Rescaling rs1(1 << 16), rs2(1 << 16);

    zkFC g = zkFC(1, embed_dim, rmsnorm_weight.weight);
    auto g_inv_rms = g(rms_inv_temp);

    auto g_inv_rms_ = rs1(g_inv_rms);

    auto GD_MLP_32_out = g_inv_rms_ * GD_outputnorm_out;

    auto v0 = ceilLog2(seq_len);
    auto v1 = ceilLog2(embed_dim);

    hadamard_product_sumcheck(g_inv_rms_, GD_outputnorm_out, random_vec(ceilLog2(GD_MLP_32_out.size)), random_vec(ceilLog2(GD_MLP_32_out.size)));
    rs1.prove(g_inv_rms, g_inv_rms_);    

    verifyWeightClaim(rmsnorm_weight, g.prove(rms_inv_temp, g_inv_rms)[0]);
    rms_inv_temp.~FrTensor();
    g_inv_rms.~FrTensor();
    g_inv_rms_.~FrTensor();

    cout << "GD for MLP_32_out successfully verified!" << endl; 


 		/*损失对自注意力输出 A_{\mathrm{out}}的梯度*/
    uint hidden_dim = 11008;


    auto up_proj = create_weight(
        workdir + "/mlp.up_proj.weight-pp.bin",
        workdir + "/" + "layer-31" + "-mlp.up_proj.weight-int.bin",
        workdir + "/" + "layer-31" + "-mlp.up_proj.weight-commitment.bin",
        embed_dim,
        hidden_dim
    );

    auto gate_proj = create_weight(
        workdir + "/mlp.gate_proj.weight-pp.bin",
        workdir + "/" + "layer-31" + "-mlp.gate_proj.weight-int.bin",
        workdir + "/" + "layer-31" + "-mlp.gate_proj.weight-commitment.bin",
        embed_dim,
        hidden_dim
    );

    auto down_proj = create_weight(
        workdir + "/mlp.down_proj.weight-pp.bin",
        workdir + "/" + "layer-31" + "-mlp.down_proj.weight-int.bin",
        workdir + "/" + "layer-31" + "-mlp.down_proj.weight-commitment.bin",
        hidden_dim,
        embed_dim
    );
    // cout << "up_proj" <<up_proj.weight.size <<endl; 
    // cout << "gate_proj" <<gate_proj.weight.size <<endl; 
    // cout << "down_proj" <<down_proj.weight.size <<endl; 

    zkFC up_layer(embed_dim, hidden_dim, up_proj.weight);
    zkFC gate_layer(embed_dim, hidden_dim, gate_proj.weight);
    zkFC down_layer(hidden_dim, embed_dim, down_proj.weight);

    Rescaling up_rescale(1 << 16);
    Rescaling gate_rescale(1 << 20);
    Rescaling hidden_rescale(1 << 16);
    Rescaling down_rescale(1 << 16);

    FrTensor swiglu_values = FrTensor::from_int_bin("swiglu-table.bin");
    FrTensor swiglu_gradient_values = FrTensor::from_int_bin("swiglu-gradient-table.bin");

    tLookupRangeMapping swiglu(-(1 << 21), 1 << 22, swiglu_values);
    tLookupRangeMapping swiglu_gradient(-(1 << 21), 1 << 22, swiglu_gradient_values);


    FrTensor Aout = FrTensor::from_int_bin(input_file_name);
    auto up_out = up_layer(Aout);
    auto up_out_ = up_rescale(up_out);
//    cout << "up_out" <<up_out.size<< endl;
//     cout << "up_proj" <<up_proj.weight.size<< endl; 

    auto gate_out = gate_layer(Aout);
    auto gate_out_ = gate_rescale(gate_out);
    auto p = swiglu(gate_out_);
    auto p_gradient = swiglu_gradient(gate_out_);

    auto &swiglu_out = p.first, &swiglu_m = p.second;
    auto &swiglu_gradient_out = p_gradient.first, &swiglu_gradient_m = p_gradient.second;

    auto temp_rand_1 = random_vec(3);
    auto swiglu_u = random_vec(ceilLog2(seq_len * hidden_dim));
    auto swiglu_v = random_vec(ceilLog2(seq_len * hidden_dim));
    vector<Polynomial> swiglu_proof, swiglu_gradient_out_proof;
    swiglu.prove(gate_out_, swiglu_out, swiglu_m, temp_rand_1[0], temp_rand_1[1], temp_rand_1[2], swiglu_u, swiglu_v, swiglu_proof);
    swiglu_gradient.prove(gate_out_, swiglu_gradient_out, swiglu_gradient_m, temp_rand_1[0], temp_rand_1[1], temp_rand_1[2], swiglu_u, swiglu_v, swiglu_gradient_out_proof);
    cout << "SwiGLU proof complete." << endl;
    vector<Polynomial> GD_swiglu_out_proof,gate_gradient_out_proof,up_gradient_out_proof;
    
    auto down_proj_weight_padded = pad_to_power_of_two(down_proj.weight,hidden_dim, embed_dim );
    uint pad_hidden_dim=pow(2, ceil(log2(hidden_dim)));
    auto GD_swiglu_out = FrTensor::matmul(GD_MLP_32_out, down_proj_weight_padded.transpose(pad_hidden_dim,embed_dim), seq_len, embed_dim, pad_hidden_dim);
   
    auto gate_gradient_in = swiglu_gradient_out * up_out_;

    auto gate_gradient_in_padded = pad_to_power_of_two(gate_gradient_in,seq_len, hidden_dim );
    auto gradient_gate = GD_swiglu_out * gate_gradient_in_padded;

    auto gate_proj_weight_padded = pad_to_power_of_two(gate_proj.weight, embed_dim,hidden_dim);
    auto gate_gradient_out = FrTensor::matmul(gradient_gate, gate_proj_weight_padded.transpose(embed_dim,pad_hidden_dim), seq_len, pad_hidden_dim, embed_dim);

    auto swiglu_out_padded = pad_to_power_of_two(swiglu_out,seq_len, hidden_dim );
    auto gradient_up = GD_swiglu_out * swiglu_out_padded;

    auto up_proj_weight_padded = pad_to_power_of_two(up_proj.weight,embed_dim, hidden_dim );
    auto up_gradient_out = FrTensor::matmul(gradient_up, up_proj_weight_padded.transpose(embed_dim,pad_hidden_dim), seq_len, pad_hidden_dim, embed_dim);
    

    auto post_att_normout = up_gradient_out + gate_gradient_out;

    auto u1_ = random_vec(ceilLog2(seq_len)); /*随机向量*/
    auto u2_ = random_vec(ceilLog2(embed_dim)); /*随机向量*/
    auto ud_ = random_vec(ceilLog2(pow(2, ceil(log2(hidden_dim)))));
    auto GD_swiglu_out_claim = GD_swiglu_out.multi_dim_me({u1_, ud_}, {seq_len, pad_hidden_dim});/*MLE*/

    auto GD_swiglu_out_final_claim = zkip(GD_swiglu_out_claim, GD_MLP_32_out.partial_me(u1_, seq_len, embed_dim), down_proj_weight_padded.transpose(pad_hidden_dim,embed_dim).partial_me(ud_,pad_hidden_dim, 1),  u2_, GD_swiglu_out_proof);  

    hadamard_product_sumcheck(swiglu_gradient_out, up_out_, random_vec(ceilLog2(gate_gradient_in.size)), random_vec(ceilLog2(gate_gradient_in.size)));

    hadamard_product_sumcheck(GD_swiglu_out, gate_gradient_in_padded, random_vec(ceilLog2(gradient_gate.size)), random_vec(ceilLog2(gradient_gate.size)));

    auto gate_gradient_out_claim = gate_gradient_out.multi_dim_me({u1_, u2_}, {seq_len, embed_dim});/*MLE*/
    auto gate_gradient_out_final_claim = zkip(gate_gradient_out_claim, gradient_gate.partial_me(u1_, seq_len, pad_hidden_dim), gate_proj_weight_padded.transpose(embed_dim,pad_hidden_dim).partial_me(u2_,embed_dim, 1),  ud_, gate_gradient_out_proof);  

    hadamard_product_sumcheck(GD_swiglu_out, swiglu_out_padded, random_vec(ceilLog2(gradient_up.size)), random_vec(ceilLog2(gradient_up.size)));

    auto up_gradient_out_claim = up_gradient_out.multi_dim_me({u1_, u2_}, {seq_len, embed_dim});/*MLE*/
    auto up_gradient_out_final_claim = zkip(up_gradient_out_claim, gradient_up.partial_me(u1_, seq_len, pad_hidden_dim), up_proj_weight_padded.transpose(embed_dim,pad_hidden_dim).partial_me(u2_,embed_dim, 1),  ud_, up_gradient_out_proof);  
    cout << "gd_post_att_normout." <<post_att_normout.size<< endl;

    post_att_normout.save("gd_post_att_normout.bin");
    cout << "GD to Att_32_out successfully verified!." << endl;


    return 0;
}


