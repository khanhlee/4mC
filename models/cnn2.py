# K-fold Cross-Validation model evaluation
from keras.models import Sequential
from keras.layers import Conv1D, Dropout, AlphaDropout, Flatten, Dense
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import KFold
from tensorflow.keras import optimizers
import numpy as np

# Assuming `inputs` and `targets` have been initialized and `num_classes` is known
# Initialize parameters for K-fold validation
kfold = KFold(n_splits=5, shuffle=True)
fold_no = 1

# Initialize lists to keep track of scores across folds
acc_per_fold = []
loss_per_fold = []

# Perform K-fold Cross-Validation
for train, test in kfold.split(inputs, targets):
    # Define the model architecture
    model = Sequential([
        Conv1D(64, 9, padding='same', input_shape=(39, 100), activation='relu'),
        Dropout(0.7),
        Conv1D(64, 9, padding='same', activation='relu'),
        Flatten(),
        AlphaDropout(0.5),
        Dense(num_classes, activation='sigmoid')
    ])
    
    # Compile the model
    model.compile(
        loss='binary_crossentropy',
        optimizer=optimizers.RMSprop(learning_rate=0.0007),
        metrics=['accuracy']
    )
    
    # Generate a print statement to keep track of fold progress
    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')
    
    # Fit data to the model
    history = model.fit(
        inputs[train], targets[train],
        batch_size=128,
        epochs=15,
        verbose=1
    )
    
    # Generate evaluation metrics
    scores = model.evaluate(inputs[test], targets[test], verbose=0)
    print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1] * 100}%')
    
    # Predict labels and generate confusion matrix
    predicted_labels = model.predict(np.stack(inputs[test]))
    cm = confusion_matrix(np.argmax(targets[test], axis=1), np.argmax(predicted_labels, axis=1))
    print('Confusion matrix:\n', cm)
    
    # Calculate AUC for training and testing sets
    pre_test_y = model.predict(inputs[test], batch_size=128)
    pre_train_y = model.predict(inputs[train], batch_size=128)
    test_auc = roc_auc_score(targets[test], pre_test_y)
    train_auc = roc_auc_score(targets[train], pre_train_y)
    print("train_auc: ", train_auc)
    print("test_auc: ", test_auc)
    
    # Record the scores for this fold
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])
    
    # Increase fold number for the next iteration
    fold_no += 1
