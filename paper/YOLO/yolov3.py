'''
1. https://pylessons.com/YOLOv3-TF2-introduction
2. https://itnext.io/implementing-yolo-v3-in-tensorflow-tf-slim-c3c55ff59dbe
'''

'''DarkNet-53  from web1'''
def DarkNet53(input_data):
	input_data = convolutional(input_data,(3,3,3,32))
	input_data = convolutional(input_data,(3,3,32,64),downsample=True)

	for i in range(1):
		input_data = residual_block(input_data,64,32,64)

	
	input_data = convolutional(input_data,(3,3,64,128),downsample=True)

	for i in range(2):
		input_data = residual_block(input_data,128,64,128)

	input_data = convolutional(input_data,(3,3,128,256),downsample=True)

	for i in range(8):
		input_data = convolutional(input_data,512,256,512)

	route_1 = input_data
	input_data = convolutional(input_data,(3,3,256,512),downsample=True)
	for i in range(8):
		input_data = convolutional(input_data,512,256,512)

	route_2 = input_data
	input_data = convolutional(input_data,(3,3,512,1024),downsample=True)

	for i in range(4):
		input_data = convolutional(input_data,1024,512,1024)
	return route_1,route_2,input_data

'''DarkNet53 from web2'''
