#include "fr-tensor.cuh"
#include "commitment.cuh"

int main(int argc, char *argv[])
{

    /*argv[1]："./zkllm-workdir/Llama-2-7b/weight-pp.bin"（生成器文件路径）。
    argv[2]："./zkllm-workdir/Llama-2-7b/layer-0-weight-int.bin"（定点化权重路径）。
    argv[3]："./zkllm-workdir/Llama-2-7b/layer-0-weight-commitment.bin"（承诺文件的保存路径）。
    argv[4]：4096（权重的行数）。
    argv[5]：5120（权重的列数）*/
    string generator_filename = argv[1];
    string param_filename = argv[2];
    string output_filename = argv[3];
    uint in_dim = std::stoi(argv[4]);
    uint out_dim = std::stoi(argv[5]);

    Commitment generator(generator_filename);
    // generator.size has to be a power of 2
    if (generator.size != (1 << ceilLog2(generator.size))) throw std::runtime_error("Generator size has to be a power of 2");


    FrTensor param = FrTensor::from_int_bin(param_filename);
    cout << "Param size: " << param.size << endl;
    cout << "In dim: " << in_dim << endl;
    cout << "Out dim: " << out_dim << endl;
    auto param_padded = param.pad({in_dim, out_dim});
    auto com = generator.commit_int(param_padded);
    cout << "Generator size: " << generator.size << endl;
    cout << "Unpadded param size: " << param.size << endl;
    cout << "Padded param size: " << param_padded.size << endl;
    cout << "Commitment size: " << com.size << endl;

    com.save(output_filename);
    return 0;
}