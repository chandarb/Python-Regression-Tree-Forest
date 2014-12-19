from regression_tree_cart import *

class Forest(object):
	def __init__(self, trees):
		self.trees = trees

	def lookup(self, x):
		"""Returns the predicted value given the parameters."""
		preds = map(lambda t: t.lookup(x), self.trees)
		return numpy.mean(preds)

	def predict_all(self, data):
		"""Returns the predicted values for a list of data points."""
		return map(lambda x: self.lookup(x), data)

def make_boot(pairs, n):
	"""Construct a bootstrap sample from the data."""
	inds = numpy.random.choice(n, size=n, replace=True)
	return dict(map(lambda x: pairs[x], inds))

def make_forest(data, B, max_depth = 500, Nmin = 5, labels = {}):
	"""Function to grow a random forest given some training data."""
	trees = []
	n = len(data)
	pairs = data.items()
	for b in range(B):
		boot = make_boot(pairs, n)
		trees.append(grow_tree(boot, 0, max_depth = max_depth, Nmin = Nmin, labels = labels, start = True, feat_bag = True))
	return Forest(trees)

