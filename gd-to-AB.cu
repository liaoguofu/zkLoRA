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
    string layer_prefix = argv[7];

    auto lm_weight = create_weight(
        workdir + "/" + "lm_head.weight-pp.bin",
        workdir + "/" + "lm_head-weight-int.bin",
        workdir + "/" + "lm_head-weight-commitment.bin",
        seq_len, vocab_size
    );
    /*logits*/

    zkFC output_layer(4096, vocab_size, lm_weight.weight);

    FrTensor gd_post_att_normout = FrTensor::from_bin("gd_post_att_normout.bin");

   auto rmsnorm_weight = create_weight(
        workdir + "/" + "model.norm.weight-pp.bin",
        workdir + "/" + "model-norm-weight-int.bin",
        workdir + "/" + "model-norm-weight-commitment.bin",
        1, embed_dim
    );    
    FrTensor rms_inv_temp = FrTensor::from_int_bin("GD_rms_inv_temp_for_0_post_attention.bin");
    vector<Polynomial> proof;


    Rescaling rs1(1 << 16), rs2(1 << 16);

    zkFC g = zkFC(1, embed_dim, rmsnorm_weight.weight);
    auto g_inv_rms = g(rms_inv_temp);

    auto g_inv_rms_ = rs1(g_inv_rms);

    auto GD_att_out = g_inv_rms_ * gd_post_att_normout;

    auto v0 = ceilLog2(seq_len);
    auto v1 = ceilLog2(embed_dim);

    hadamard_product_sumcheck(g_inv_rms_, GD_att_out, random_vec(ceilLog2(GD_att_out.size)), random_vec(ceilLog2(GD_att_out.size)));
    rs1.prove(g_inv_rms, g_inv_rms_);    

    verifyWeightClaim(rmsnorm_weight, g.prove(rms_inv_temp, g_inv_rms)[0]);

    FrTensor softmax_out = FrTensor::from_bin("layer-31_softmax_out.bin");

    auto GD_att_v_out = FrTensor::matmul( softmax_out.transpose(seq_len,seq_len),GD_att_out, seq_len,seq_len, embed_dim);



    FrTensor initial_A = FrTensor::from_int_bin("initial_A.bin");
    FrTensor initial_B = FrTensor::from_int_bin("initial_B.bin");
    FrTensor attn_input = FrTensor::from_int_bin("attn_input.bin");

    auto BX = FrTensor::matmul(attn_input, initial_B, seq_len,embed_dim, 8);
    auto GD_B = FrTensor::matmul(BX.transpose(seq_len,8),GD_att_v_out,  8,seq_len, embed_dim);

    auto u1 = random_vec(ceilLog2(seq_len)); /*随机向量*/
    auto u2 = random_vec(ceilLog2(embed_dim)); /*随机向量*/
    auto u3 = random_vec(ceilLog2(8)); /*随机向量*/
    auto ud = random_vec(ceilLog2(pow(2, ceil(log2(vocab_size)))));

    auto claim = GD_att_v_out.multi_dim_me({u1, u2}, {seq_len, embed_dim});/*MLE*/

    auto final_claim = zkip(claim,softmax_out.transpose(seq_len,seq_len).partial_me(u1, seq_len, seq_len), GD_att_out.partial_me(u2,embed_dim, 1),  u1, proof);  
   
    cout << "GD to Att_32_V_out successfully verified!." << endl;
    auto BX_claim = BX.multi_dim_me({u1, u3}, {seq_len, 8});/*MLE*/
    auto BX_final_claim = zkip(BX_claim,attn_input.partial_me(u1, seq_len, embed_dim), initial_B.partial_me(u3,8, 1),  u2, proof);  
   
    auto GD_B_claim = GD_B.multi_dim_me({u3, u2}, {8, embed_dim});/*MLE*/
    auto GD_B_final_claim = zkip(GD_B_claim,BX.transpose(seq_len,8).partial_me(u3, 8, seq_len), GD_att_v_out.partial_me(u2,embed_dim, 1),  u1, proof);     
    cout << "GD to A successfully verified!." << endl;

    auto temp = FrTensor::matmul(attn_input.transpose(seq_len,embed_dim), GD_att_v_out, embed_dim,seq_len, embed_dim);
    auto GD_A = FrTensor::matmul(temp,initial_A.transpose(8,embed_dim),  embed_dim,embed_dim, 8);

    auto temp_claim = temp.multi_dim_me({u2, u2}, {embed_dim, embed_dim});/*MLE*/
    auto temp_final_claim = zkip(temp_claim,attn_input.transpose(seq_len,embed_dim).partial_me(u2, embed_dim, seq_len), GD_att_v_out.partial_me(u2,embed_dim, 1),  u1, proof);  
   
    auto GD_A_claim = GD_A.multi_dim_me({u2, u3}, {embed_dim, 8});/*MLE*/
    auto GD_A_final_claim = zkip(GD_A_claim,temp.partial_me(u2, embed_dim, embed_dim), initial_A.transpose(8,embed_dim).partial_me(u3,8, 1),  u2, proof);  
   
    cout << "GD to B successfully verified!." << endl;


    return 0;
}


