#include <torch/torch.h>
#include <iostream>

int main() {

  /*
  torch::nn::Linear model(num_features, 1);
  torch::optim::SGD optimizer(model.parameters());
  //torch::data::DataLoader data_loader(dataset);

  for(size_t epoch = 0; epoch < 10; ++epoch) {
    for(auto [example, label] : data_loader) {
      auto prediction = model->forward(example);
      auto loss = loss_function(prediction, label);
      loss.backward();
      optimizer.step();
    }
  }
  */


  torch::Tensor tensor = torch::rand({2, 3});
  std::cout << tensor << std::endl;
}