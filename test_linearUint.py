
class LinearUint:
	'''
	'''
	def __init__(self, vec_num, fun):
		self.weights = 0.0
		self.bias = 0.0
		self.activator = fun
	def train_data(self, input_vec, input_labels, delta, iteators):
		'''
		'''
		for i in range(iteators):
			self.train(input_vec, input_labels, delta)
			print(self.weights, self.bias)

	def train(self, input_vec, input_labels, delta):
		'''
		'''
		#predict
		for i in range(len(input_vec)):
			one_vec = input_vec[i]
			one_label = input_labels[i]
			one_pred = self.predict(one_vec)

			self.update_weights(one_vec, one_pred, one_label, delta)


	def predict(self, one_vec):
		'''
		'''
		value = self.weights * one_vec + self.bias
		out_pred = self.activator(value)
		return out_pred

	def update_weights(self, one_vec, one_pred, one_label, delta):
		''''
		'''
		diff = one_label - one_pred
		delta_w = delta * diff * one_vec
		#update 
		self.weights = self.weights + delta_w
		self.bias = self.bias + diff * delta


def fun(x):
	return x

def getTrainData():
	''''
	'''
	input_vecs = [5, 3, 8, 1.4, 10.1]
	labels = [5500, 2300, 7600, 1800, 11400]
	return input_vecs, labels



linearUint = LinearUint(1, fun)
input_vecs, input_labels = getTrainData()
linearUint.train_data(input_vecs, input_labels, 0.01, 10)


print 'Work 3.4 years, monthly salary = %.2f' % linearUint.predict(3.4)
print 'Work 15 years, monthly salary = %.2f' % linearUint.predict(15)
print 'Work 1.5 years, monthly salary = %.2f' % linearUint.predict(1.5)
print 'Work 6.3 years, monthly salary = %.2f' % linearUint.predict(6.3)

