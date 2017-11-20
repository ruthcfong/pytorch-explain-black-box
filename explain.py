import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models
import torch.utils.model_zoo as model_zoo
import cv2
import sys
import numpy as np

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
Tensor = FloatTensor

#alexnet definition that conveniently let's you grab the outputs from any layer. 
#Also we ignore dropout here
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        #convolutional layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
        self.conv2 = nn.Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        self.conv3 = nn.Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv4 = nn.Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv5 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #pooling layers
        self.pool1 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), dilation=(1, 1))
        self.pool2 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), dilation=(1, 1))
        self.pool5 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), dilation=(1, 1))
        #fully connected layers
        self.fc1 = nn.Linear(9216, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 1000)
        
    def forward(self, x, out_keys = None):
        out = {}
        out['c1'] = self.conv1(x)
        out['r1'] = F.relu(out['c1'])
        out['p1'] = self.pool1(out['r1'])
        out['r2'] = F.relu(self.conv2(out['p1']))
        out['p2'] = self.pool2(out['r2'])
        out['r3'] = F.relu(self.conv3(out['p2']))
        out['r4'] = F.relu(self.conv4(out['r3']))
        out['r5'] = F.relu(self.conv5(out['r4']))
        out['p5'] = self.pool5(out['r5'])
        out['fc1'] = F.relu(self.fc1(out['p5'].view(1, -1)))
        out['fc2'] = F.relu(self.fc2(out['fc1']))
        out['fc3'] = self.fc3(out['fc2'])

        if out_keys is None:
            return out['fc3']

        res = {}
        for key in out_keys:
            res[key] = out[key]
        return res
        #return [out[key] for key in out_keys]


def alexnet(pretrained, **kwargs):
    model = AlexNet(**kwargs)
    if pretrained:
        keys = model.state_dict().keys()
        new_weights = {}
        weights = model_zoo.load_url(model_urls['alexnet'])
        for k, key in enumerate(weights.keys()):
            new_weights[keys[k]] = weights[key]
        model.load_state_dict(new_weights)
    return model


def min_norm(input_dict, target_dict):
    assert(np.array_equal(input_dict.keys(), target_dict.keys()))
    loss = 0
    for key in input_dict.keys():
        loss += torch.mean(torch.clamp(input_dict[key]-target_dict[key], min=0))
    return loss


def tv_norm(input, tv_beta):
	img = input[0, 0, :]
	row_grad = torch.mean(torch.abs((img[:-1 , :] - img[1 :, :])).pow(tv_beta))
	col_grad = torch.mean(torch.abs((img[: , :-1] - img[: , 1 :])).pow(tv_beta))
	return row_grad + col_grad


def preprocess_image(img):
	means=[0.485, 0.456, 0.406]
	stds=[0.229, 0.224, 0.225]

	preprocessed_img = img.copy()[: , :, ::-1]
	for i in range(3):
		preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
		preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
	preprocessed_img = \
		np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))

	if use_cuda:
		preprocessed_img_tensor = torch.from_numpy(preprocessed_img).cuda()
	else:
		preprocessed_img_tensor = torch.from_numpy(preprocessed_img)

	preprocessed_img_tensor.unsqueeze_(0)
	return Variable(preprocessed_img_tensor, requires_grad = False)


def save(mask, img, blurred):
	mask = mask.cpu().data.numpy()[0]
	mask = np.transpose(mask, (1, 2, 0))

	mask = (mask - np.min(mask)) / np.max(mask)
	mask = 1 - mask
	heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
	
	heatmap = np.float32(heatmap) / 255
	cam = 1.0*heatmap + np.float32(img)/255
	cam = cam / np.max(cam)

	img = np.float32(img) / 255
	perturbated = np.multiply(1 - mask, img) + np.multiply(mask, blurred)	

	cv2.imwrite("perturbated.png", np.uint8(255*perturbated))
	cv2.imwrite("heatmap.png", np.uint8(255*heatmap))
	cv2.imwrite("mask.png", np.uint8(255*mask))
	cv2.imwrite("cam.png", np.uint8(255*cam))

def numpy_to_torch(img, requires_grad = True):
	if len(img.shape) < 3:
		output = np.float32([img])
	else:
		output = np.transpose(img, (2, 0, 1))

	output = torch.from_numpy(output)
	if use_cuda:
		output = output.cuda()

	output.unsqueeze_(0)
	v = Variable(output, requires_grad = requires_grad)
	return v


def load_model():
	#model = models.vgg19(pretrained=True)
        model = alexnet(pretrained=True)
	model.eval()
	if use_cuda:
		model.cuda()
	
        for p in model.parameters():
            p.requires_grad = False
	#for p in model.features.parameters():
	#	p.requires_grad = False
 	#for p in model.classifier.parameters():
        #    p.requires_grad = False

	return model


if __name__ == '__main__':
	#Hyper parameters. 
	#TBD: Use argparse
	tv_beta = 3
	learning_rate = 0.1
	max_iterations = 500
	l1_coeff = 0.01
	tv_coeff = 0.2
        less_coeff = 0.5#1e-3 

	model = load_model()
	original_img = cv2.imread(sys.argv[1], 1)
	original_img = cv2.resize(original_img, (224, 224))
	img = np.float32(original_img) / 255
	blurred_img1 = cv2.GaussianBlur(img, (11, 11), 5)
	blurred_img2 = np.float32(cv2.medianBlur(original_img, 11))/255
	blurred_img_numpy = (blurred_img1 + blurred_img2) / 2
	mask_init = np.ones((28, 28), dtype = np.float32)
	
	# Convert to torch variables
	img = preprocess_image(img)
	blurred_img = preprocess_image(blurred_img2)
	mask = numpy_to_torch(mask_init)

	if use_cuda:
		upsample = torch.nn.UpsamplingBilinear2d(size=(224, 224)).cuda()
	else:
		upsample = torch.nn.UpsamplingBilinear2d(size=(224, 224))
	optimizer = torch.optim.Adam([mask], lr=learning_rate)

        #out_keys = ['p1','p2','r3','r4','p5','fc1','fc2','fc3']
        out_keys = ['p5']

        target_res = model(img, out_keys)
	target = torch.nn.Softmax()(model(img))
	category = np.argmax(target.cpu().data.numpy())
	print "Category with highest probability", category
	print "Optimizing.. "


	for i in range(max_iterations):
		upsampled_mask = upsample(mask)
		# The single channel mask is used with an RGB image, 
		# so the mask is duplicated to have 3 channel,
		upsampled_mask = \
			upsampled_mask.expand(1, 3, upsampled_mask.size(2), \
										upsampled_mask.size(3))
		
		# Use the mask to perturbated the input image.
		perturbated_input = img.mul(upsampled_mask) + \
							blurred_img.mul(1-upsampled_mask)
		
		noise = np.zeros((224, 224, 3), dtype = np.float32)
		noise = noise + cv2.randn(noise, 0, 0.2)
		noise = numpy_to_torch(noise)
		perturbated_input = perturbated_input + noise
		
                res = model(perturbated_input, out_keys)
		outputs = torch.nn.Softmax()(model(perturbated_input))
                l1_loss = l1_coeff*torch.mean(torch.abs(1-mask))
                tv_loss = tv_coeff*tv_norm(mask, tv_beta)
                less_loss = less_coeff*min_norm(res, target_res)
                class_loss = outputs[0, category]
		tot_loss = l1_loss + tv_loss + less_loss + class_loss

		optimizer.zero_grad()
		tot_loss.backward()
		optimizer.step()

		# Optional: clamping seems to give better results
		mask.data.clamp_(0, 1)

                if i % 25 == 0:
                    print('Epoch %d\tL1 Loss %f\tTV Loss %f\tLess Loss %f\tClass Loss %f\tTot Loss %f\t' 
                            % (i+1, l1_loss.data.cpu().numpy()[0], tv_loss.data.cpu().numpy()[0],
                                less_loss.data.cpu().numpy()[0], class_loss.data.cpu().numpy()[0],
                                tot_loss.data.cpu().numpy()[0]))

	upsampled_mask = upsample(mask)
	save(upsampled_mask, original_img, blurred_img_numpy)
