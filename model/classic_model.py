#from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def get_model(model_type):
	
	if model_type == 'bayes':
		model = MultinomialNB()
	elif model_type == 'svm':
		model = SVC()
	elif model_type == 'mlp':
		model = MLPClassifier()
	elif model_type == 'decision_tree':
		model = DecisionTreeClassifier()
	elif model_type == 'random_forest':
		model = RandomForestClassifier()

	return model
