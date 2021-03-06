import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler


from deep_and_cross import build_deep_and_cross




def reverse_onehot(df):
    """ Reverse a onehoted dataframe into a single series """
    return df.dot(np.array(range(df.shape[1]))).astype(int)


def prepare():
	df = pd.read_csv('./data/covtype.csv')

	target_col = 'Cover_Type'
	continuous_cols = [c for c in df if c != target_col and 'Soil' not in c and 'Wilderness' not in c]
	df[continuous_cols] = StandardScaler().fit_transform(df[continuous_cols])	

	df['soil_type'] = reverse_onehot(df[[c for c in df if 'Soil' in c]])
	df['wilderness_area'] = reverse_onehot(df[[c for c in df if 'Wilderness' in c]])

	discrete_feature_size = {
	    'soil_type': df['soil_type'].nunique(),
	    'wilderness_area': df['wilderness_area'].nunique()
	}

	continuous_feature_size = len(continuous_cols)


	y = pd.get_dummies(df[target_col])
	train_idx, test_idx = next(StratifiedShuffleSplit(test_size=0.1, random_state=1024).split(df, y))

	X_train = {
		     'soil_type': df.loc[train_idx, 'soil_type'].values,
		     'wilderness_area': df.loc[train_idx, 'wilderness_area'].values,
		     'continuous_features': df.loc[train_idx, continuous_cols].values.astype(float), 
	         }

	X_test = {
		     'soil_type': df.loc[test_idx, 'soil_type'].values,
		     'wilderness_area': df.loc[test_idx, 'wilderness_area'].values,
		     'continuous_features': df.loc[test_idx, continuous_cols].values.astype(float), 
	         }

	y_train = y.loc[train_idx, :].values
	y_test = y.loc[test_idx, :].values

	return discrete_feature_size, continuous_feature_size, X_train, y_train, X_test, y_test


discrete_feature_size, continuous_feature_size, X_train, y_train, X_test, y_test = prepare()


# The optimal hyperparameter settings in the original paper 
# were 8 cross layers of size 54 and 6 deep layers of size 292 for DCN
DCN = build_deep_and_cross(discrete_feature_size, continuous_feature_size, 
							label_size=y_train.shape[1],
							n_deep_layers=6,
							deep_layer_size=100,
							n_cross_layers=8,
							share_weights=False)
DCN.compile(optimizer=tf.optimizers.Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
DCN.fit(x=X_train, y=y_train, validation_data=(X_test, y_test),
		batch_size=256, epochs=20, shuffle=True)

