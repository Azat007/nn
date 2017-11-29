import numpy as np
import matplotlib.pyplot as plt
import time


class Neuralnet:
	def __init__(self, neurons):
		self.layers = len(neurons)

		# Learning rate
		self.rate = 0.01

		# Input вектор
		self.inputs = []
		# Output вектор
		self.outputs = []
		# Error вектор
		self.errors = []
		# Weight (матрица весов)
		self.weights = []
		# Bias вектор (вектор смещения)
		self.biases = []

		for layer in range(self.layers):
			# Создаём input, output, and error вектора
			self.inputs.append(np.empty(neurons[layer]))
			self.outputs.append(np.empty(neurons[layer]))
			self.errors.append(np.empty(neurons[layer]))

		for layer in range(self.layers - 1):
			# Создаём  weight
			self.weights.append(np.random.normal(
				scale=0.8/np.sqrt(neurons[layer]),
				size=[neurons[layer], neurons[layer + 1]]
			))
			# Создаём bias вектор
			self.biases.append(np.random.normal(
				scale=0.2/np.sqrt(neurons[layer]),
				size=neurons[layer + 1]
			))

	def feedforward(self, inputs):
		# Set input neuron inputs
		self.inputs[0] = inputs
		for layer in range(self.layers - 1):
			# Find output of this layer from its input
			self.outputs[layer] = np.tanh(self.inputs[layer])
			# Find input of next layer from output of this layer and weight matrix (plus bias)
			self.inputs[layer + 1] = np.dot(self.weights[layer].T, self.outputs[layer]) + self.biases[layer]
		self.outputs[-1] = np.tanh(self.inputs[-1])

	def backpropagate(self, targets):
		# Calculate error at output layer
		self.errors[-1] = self.outputs[-1] - targets
		# Calculate error vector for each layer
		for layer in reversed(range(self.layers - 1)):
			gradient = 1 - self.outputs[layer] * self.outputs[layer]
			self.errors[layer] = gradient * np.dot(self.weights[layer], self.errors[layer + 1])
		# Adjust weight matrices and bias vectors
		for layer in range(self.layers - 1):
			self.weights[layer] -= self.rate * np.outer(self.outputs[layer], self.errors[layer + 1])
			self.biases[layer] -= self.rate * self.errors[layer + 1]


# Create a neural network that accepts a 28 by 28 array as input and has 10 output neurons
net = Neuralnet([28 * 28, 10, 10])

# Extract handwritten digits from files
digits = []
for digit in range(10):
	with open('digits/' + str(digit), 'r') as digitfile:
		digits.append(np.fromfile(digitfile, dtype=np.uint8).reshape(1000, 28, 28))

# Train neural network on entire data set multiple times
begin_time = time.time()
for epoch in range(10):
	# Total error for this epoch
	error = 0
	# Choose a sample index
	for sample in np.random.permutation(1000):
		# Choose a digit
		for digit in np.random.permutation(10):
			# Extract input data
			inputs = digits[digit][sample].flatten()
			# Feed input data to neural network
			net.feedforward(inputs)
			# Target output consists of -1s except for matching digit
			targets = np.full(10, -1, dtype=np.float32)
			targets[digit] = 1
			# Train neural network based on target output
			net.backpropagate(targets)
			error += np.sum(net.errors[-1] * net.errors[-1])
	print ('Epoch ' + str(epoch) + ' error: ' + str("%.3f" % (error/1000)))
end_time = time.time()
print('-------------------------')
print ('Время обучения: ' + str("%.2f" % (end_time - begin_time))+ ' сек.')
print('-------------------------')


i=True

digittest = []
for digit in range(1):
	with open('digettest/' + str(digit), 'r') as digitfile:
		digittest.append(np.fromfile(digitfile, dtype=np.uint8).reshape(10000, 28, 28))

def get_array():
    import pyperclip
    return pyperclip.paste()


while i:
	inputstring = input('Введите 0 (выход-10): ')
	if inputstring.isdigit():
		digit = int(inputstring)
		if digit==10:
			i=False
		if digit in range(1):
			# Choose a random sample
			sample = np.random.randint(10000)
			image = digittest[digit][sample]
			# Show image being fed into neural network
			plt.imshow(image, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
			plt.show()

			# Feed image into neural network
			net.feedforward(image.flatten())

			# Print neural network outputs and classification
			print ('Распознано как: ' + str(np.argmax(net.outputs[-1])))

def get_array():
    import pyperclip
    return pyperclip.paste()
s=True
while s:
	print('--------------------')
	print("Нажмите enter")
	print("(Выход -[10]")
	inp = input()
	if inp == "10":
		break
	a = get_array()
	a = a.split(',')
	b = []
	for i in a:
		b.append(int(i))
	b=np.array(b)
	b.shape=(28,28)
	plt.imshow(b, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
	plt.show()
	net.feedforward(b.flatten())

	# Print neural network outputs and classification
	print('Распознано как: ' + str(np.argmax(net.outputs[-1])))


