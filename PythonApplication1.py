
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
# from recsys_utils import *
# from recsys.algorithm.factorize import SVD

def load_precalc_params_small():

    file = open('C:/Users/ASUS/OneDrive/Desktop/Collaborating-Filtering/small_movies_X.csv', 'rb')
    X = np.loadtxt(file, delimiter = ",")

    file = open('C:/Users/ASUS/OneDrive/Desktop/Collaborating-Filtering/small_movies_W.csv', 'rb')
    W = np.loadtxt(file, delimiter = ",")

    file = open('C:/Users/ASUS/OneDrive/Desktop/Collaborating-Filtering/small_movies_b.csv', 'rb')
    b = np.loadtxt(file, delimiter = ",")
    b = b.reshape(1,-1)
    num_movies, num_features = X.shape
    num_users,_ = W.shape
    return (X, W, b, num_movies, num_features, num_users)

def load_ratings_small():
    file = open('C:/Users/ASUS/OneDrive/Desktop/Collaborating-Filtering/small_movies_Y.csv', 'rb')
    Y = np.loadtxt(file,delimiter = ",")

    file = open('C:/Users/ASUS/OneDrive/Desktop/Collaborating-Filtering/small_movies_R.csv', 'rb')
    R = np.loadtxt(file,delimiter = ",")
    return(Y,R)

X, W, b, num_movies, num_features, num_users = load_precalc_params_small()
Y, R = load_ratings_small()

print("Y", Y.shape, "R", R.shape)
print("X", X.shape)
print("W", W.shape)
print("b", b.shape)
print("num_features", num_features)
print("num_movies",   num_movies)
print("num_users",    num_users)

tsmean =  np.mean(Y[0, R[0, :].astype(bool)])
print(f"Average rating for movie 1 : {tsmean:0.3f} / 5" )

##This code defines the cost function for a content-based filtering system, 
#specifically a collaborative filtering model.
# The function calculates the cost (or error) of the predicted movie ratings compared to the actual ratings, 

def cofi_cost_func(X, W, b, Y, R, lambda_):
    """
    Returns the cost for the content-based filtering
    Args:
      X (ndarray (num_movies,num_features)): matrix of item features
      W (ndarray (num_users,num_features)) : matrix of user parameters
      b (ndarray (1, num_users)            : vector of user parameters
      Y (ndarray (num_movies,num_users)    : matrix of user ratings of movies
      R (ndarray (num_movies,num_users)    : matrix, where R(i, j) = 1 if the i-th movies was rated by the j-th user
      lambda_ (float): regularization parameter
    Returns:
      J (float) : Cost
    """
    nm, nu = Y.shape
    J = 0
    ### START CODE HERE ###
    for j in range(nu):
        w = W[j, :]
        b_j = b[0, j]
        for i in range(nm):
            x = X[i, :]
            y = Y[i, j]
            r = R[i, j]
            J += np.square(r * (np.dot(w, x) + b_j - y))
    J = J / 2
    ## REGULARIZATION
        # J += (lambda_/2) * (np.sum(np.square(W)) + np.sum(np.square(X)))
    #   A regularization parameter that helps prevent overfitting by penalizing large values of the model parameters
    ##
    ### END CODE HERE ###
    return J


print(f"################################################ cost with small dataset ################################################ ")
#Example with small dataset

cost_error_actual = cofi_cost_func(X, W, b, Y, R, 1)

print(cost_error_actual)

num_users_r = 4
num_movies_r = 5 
num_features_r = 3

X_r = X[:num_movies_r, :num_features_r]
W_r = W[:num_users_r,  :num_features_r]
b_r = b[0, :num_users_r].reshape(1,-1)
Y_r = Y[:num_movies_r, :num_users_r]
R_r = R[:num_movies_r, :num_users_r]

# Evaluate cost function
J = cofi_cost_func(X_r, W_r, b_r, Y_r, R_r, 0);
print(f"Cost With small dataset: {J:0.2f}")



print(f"################################################ Use Vectorized Implementation Of Cost Function  ################################################ ")

def cofi_cost_func_vectorized_version(X, W, b, Y, R, lambda_):
    """
    Returns the cost for the content-based filtering
    Vectorized for speed. Uses tensorflow operations to be compatible with custom training loop.
    Args:
      X (ndarray (num_movies,num_features)): matrix of item features
      W (ndarray (num_users,num_features)) : matrix of user parameters
      b (ndarray (1, num_users)            : vector of user parameters
      Y (ndarray (num_movies,num_users)    : matrix of user ratings of movies
      R (ndarray (num_movies,num_users)    : matrix, where R(i, j) = 1 if the i-th movies was rated by the j-th user
      lambda_ (float): regularization parameter
    Returns:
      J (float) : Cost
    """
    j = (tf.linalg.matmul(X, tf.transpose(W)) + b - Y)*R
    J = 0.5 * tf.reduce_sum(j**2) + (lambda_/2) * (tf.reduce_sum(X**2) + tf.reduce_sum(W**2))
    return J


# Evaluate cost function
J = cofi_cost_func_vectorized_version(X_r, W_r, b_r, Y_r, R_r, 0);
print(f"Cost: {J:0.2f}")

# Evaluate cost function with regularization 
J = cofi_cost_func_vectorized_version(X_r, W_r, b_r, Y_r, R_r, 1.5);
print(f"Cost (with regularization): {J:0.2f}")

print(f"################################################ Predictions ################################################ ")

# def load_Movie_List_pd():
#     file = open('C:/Users/ASUS/OneDrive/Desktop/Collaborating-Filtering/small_movie_list.csv', 'rb')
#     rtn = np.loadtxt(file, delimiter = ",")
    
#     return rtn



def normalizeRatings(Y, R):

    num_movies, num_users = Y.shape
    Ymean = np.zeros((num_movies, 1))
    Ynorm = np.zeros(Y.shape)

    for i in range(num_movies):
        # Compute mean rating for each movie that has been rated
        rated_idx = np.where(R[i, :] == 1)
        Ymean[i] = np.mean(Y[i, rated_idx])
        
        Ynorm[i, rated_idx] = Y[i, rated_idx] - Ymean[i]
    
    return Ynorm, Ymean


def load_Movie_List_pd():
    # Use pandas to load the CSV file
    file = open('C:/Users/ASUS/OneDrive/Desktop/Collaborating-Filtering/small_movie_list.csv', 'rb')
    df = pd.read_csv(file, delimiter=",")  # Customize `header=None` if no header row
    return df


movieList = load_Movie_List_pd()

my_ratings = np.zeros(num_movies)          #  Initialize my ratings

# Check the file small_movie_list.csv for id of each movie in our dataset
# For example, Toy Story 3 (2010) has ID 2700, so to rate it "5", you can set
my_ratings[2700] = 5 

#Or suppose you did not enjoy Persuasion (2007), you can set
my_ratings[2609] = 2;

# We have selected a few movies we liked / did not like and the ratings we
# gave are as follows:
my_ratings[929]  = 5   # Lord of the Rings: The Return of the King, The
my_ratings[246]  = 5   # Shrek (2001)
my_ratings[2716] = 3   # Inception
my_ratings[1150] = 5   # Incredibles, The (2004)
my_ratings[382]  = 2   # Amelie (Fabuleux destin d'Amélie Poulain, Le)
my_ratings[366]  = 5   # Harry Potter and the Sorcerer's Stone (a.k.a. Harry Potter and the Philosopher's Stone) (2001)
my_ratings[622]  = 5   # Harry Potter and the Chamber of Secrets (2002)
my_ratings[988]  = 3   # Eternal Sunshine of the Spotless Mind (2004)
my_ratings[2925] = 1   # Louis Theroux: Law & Disorder (2008)
my_ratings[2937] = 1   # Nothing to Declare (Rien à déclarer)
my_ratings[793]  = 5   # Pirates of the Caribbean: The Curse of the Black Pearl (2003)
my_rated = [i for i in range(len(my_ratings)) if my_ratings[i] > 0]

print('\nNew user ratings:\n')
for i in range(len(my_ratings)):
    if my_ratings[i] > 0 :
        print(f'Rated {my_ratings[i]} for  {movieList.loc[i,"title"]}');


#Now, let's add these reviews to  𝑌and  𝑅 and normalize the ratings.

# Reload ratings
Y, R = load_ratings_small()

# Add new user ratings to Y 
Y = np.c_[my_ratings, Y]

# Add new user indicator matrix to R
R = np.c_[(my_ratings != 0).astype(int), R]

# Normalize the Dataset
Ynorm, Ymean = normalizeRatings(Y, R)


#  Useful Values
num_movies, num_users = Y.shape
num_features = 100

# Set Initial Parameters (W, X), use tf.Variable to track these variables
tf.random.set_seed(1234) # for consistent results
W = tf.Variable(tf.random.normal((num_users,  num_features),dtype=tf.float64),  name='W')
X = tf.Variable(tf.random.normal((num_movies, num_features),dtype=tf.float64),  name='X')
b = tf.Variable(tf.random.normal((1,          num_users),   dtype=tf.float64),  name='b')

# Instantiate an optimizer.
optimizer = keras.optimizers.Adam(learning_rate=1e-1)

iterations = 200
lambda_ = 1
for iter in range(iterations):
    # Use TensorFlow’s GradientTape
    # to record the operations used to compute the cost 
    with tf.GradientTape() as tape:

        # Compute the cost (forward pass included in cost)
        cost_value = cofi_cost_func_vectorized_version(X, W, b, Ynorm, R, lambda_)

    # Use the gradient tape to automatically retrieve
    # the gradients of the trainable variables with respect to the loss
    grads = tape.gradient( cost_value, [X,W,b] )

    # Run one step of gradient descent by updating
    # the value of the variables to minimize the loss.
    optimizer.apply_gradients( zip(grads, [X,W,b]) )

    # Log periodically.
    if iter % 20 == 0:
        print(f"Training loss at iteration {iter}: {cost_value:0.1f}")
        


#Final Step For Predictions
        
# Make a prediction using trained weights and biases
p = np.matmul(X.numpy(), np.transpose(W.numpy())) + b.numpy()

#restore the mean
pm = p + Ymean

my_predictions = pm[:,0]

# sort predictions
ix = tf.argsort(my_predictions, direction='DESCENDING')

for i in range(17):
    j = ix[i]
    if j not in my_rated:
        print(f'Predicting rating {my_predictions[j]:0.2f} for movie {movieList.loc[i,"title"]}')

print('\n\nOriginal vs Predicted ratings:\n')
for i in range(len(my_ratings)):
    if my_ratings[i] > 0:
        print(f'Original {my_ratings[i]}, Predicted {my_predictions[i]:0.2f} for {movieList.loc[i,"title"]}')        
        

filter=(movieList["number of ratings"] > 20)
movieList["pred"] = my_predictions
movieList_df = movieList.reindex(columns=["pred", "mean rating", "number of ratings", "title"])
movieList_df.loc[ix[:300]].loc[filter].sort_values("mean rating", ascending=False)
