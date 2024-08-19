import torch


def cal_model_flops(model, input):
     model.eval()
     input = torch.ones(input, dtype = torch.float32).cuda()
     flops_list=[]
     def conv_hook(self, input, output):
         output_channels, output_height, output_width = output[0].size()
         flops = (self.out_channels/self.groups) * (self.kernel_size[0] * self.kernel_size[1] *self.in_channels/self.groups) * output_height * output_width*self.groups
         flops_list.append(flops)

     def linear_hook(self, input, output):
         flops = self.in_features*self.out_features
         flops_list.append(flops)

     def foo(net):
         childrens = list(net.children())
         if not childrens:
            if isinstance(net, torch.nn.Conv2d):
                net.register_forward_hook(conv_hook)
            if isinstance(net, torch.nn.Linear):
                net.register_forward_hook(linear_hook)
            return
         for c in childrens:
            foo(c)
     foo(model)
     output = model(input)
     print(flops_list)
     flops = sum(flops_list)
     return flops