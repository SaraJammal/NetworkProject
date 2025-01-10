# # # # import os
# # # # import pandas as pd
# # # # from keras import Sequential
# # # # from keras.src.layers import LSTM, Dropout, Dense
# # # #
# # # # from sklearn.model_selection import train_test_split
# # # # from sklearn.preprocessing import MinMaxScaler, LabelEncoder
# # # #
# # # # for dirname, _, filenames in os.walk(r'C:\Users\USER\Desktop\network'):
# # # #     print("dir name",dirname)
# # # #     for filename in filenames:
# # # #         print(os.path.join(dirname, filename))
# # # #
# # # # #
# # # # # # Install TensorFlow in your Python environment using: pip install tensorflow
# # # # # from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
# # # # # from sklearn.metrics import confusion_matrix, classification_report
# # # # # import matplotlib.pyplot as plt
# # # # # from sklearn.metrics import roc_curve, auc
# # # # # import numpy as np
# # # #
# # # # def merge_files_in_folder(folder_path):
# # # #     # Initialize an empty list to store DataFrames
# # # #     dfs = []
# # # #
# # # #     # Loop through each file in the folder
# # # #     for filename in os.listdir(folder_path):
# # # #         # Extract label from filename and remove numbers at the end
# # # #         label = filename.split("_train.pcap.csv")[0].split("_test.pcap.csv")[0].split("1234567890")[0]
# # # #         # Remove numeric part if it exists
# # # #         label = ''.join(filter(str.isalpha, label))
# # # #         # Read the file into a DataFrame
# # # #         filepath = os.path.join(folder_path, filename)
# # # #         df = pd.read_csv(filepath)
# # # #
# # # #         # Add a 'label' column with the current label
# # # #         df['label'] = label
# # # #
# # # #         # Append the DataFrame to the list
# # # #         dfs.append(df)
# # # #
# # # #     # Concatenate all DataFrames in the list
# # # #     merged_data = pd.concat(dfs, ignore_index=True)
# # # #
# # # #     # Display value counts of labels
# # # #     print("Value counts of labels in train data:")
# # # #     print(merged_data['label'].value_counts())
# # # #     return merged_data
# # # #
# # # #
# # # # # Merge files in the train folder
# # # # train_folder_path = r"C:\Users\20238023\Downloads\network\network\train dataset"
# # # # test_folder_path = r"C:\Users\20238023\Downloads\network\network\test dataset"
# # # #
# # # # train_df = merge_files_in_folder(train_folder_path)
# # # # test_df = merge_files_in_folder(test_folder_path)
# # # #
# # # #
# # # #
# # # # # Assuming the target variable is 'label'
# # # # X = train_df.drop(columns=['label'])
# # # # y = train_df['label']
# # # #
# # # # # Encode the labels if they are categorical
# # # # label_encoder = LabelEncoder()
# # # # y = label_encoder.fit_transform(y)
# # # #
# # # # # Normalize features
# # # # scaler = MinMaxScaler()
# # # # X = scaler.fit_transform(X)
# # # #
# # # # # Reshape X for LSTM (samples, timesteps, features)
# # # # X = X.reshape(X.shape[0], 1, X.shape[1])
# # # #
# # # # # Split into training and test sets
# # # # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# # # #
# # # # # Step 2: Define the LSTM model
# # # # model = Sequential([
# # # #     LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
# # # #     Dropout(0.2),
# # # #     LSTM(64, return_sequences=False),
# # # #     Dropout(0.2),
# # # #     Dense(32, activation='relu'),
# # # #     Dense(1, activation='sigmoid')  # Use 'softmax' if there are multiple classes
# # # # ])
# # # #
# # # # # Step 3: Compile the model
# # # # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# # # #
# # # # # Step 4: Train the model
# # # # history = model.fit(X_train, y_train, epochs=20, batch_size=64, validation_split=0.2)
# # # #
# # # # # Step 5: Evaluate the model
# # # # loss, accuracy = model.evaluate(X_test, y_test)
# # # # print(f"Test Loss: {loss}")
# # # # print(f"Test Accuracy: {accuracy}")
# # # #
# # # # # Optional: Save the trained model
# # # # model.save('lstm_iomt_model.h5')
# # # #
# # # #
# # # # # Mapping labels to new categories for 6 classes
# # # # label_map_6_classes = {
# # # #     "DDos": ["TCPIPDDoSUDP", "TCPIPDDoSICMP", "TCPIPDDoSTCP", "TCPIPDDoSSYN"],
# # # #     "Dos": ["TCPIPDoSUDP", "TCPIPDoSSYN", "TCPIPDoSICMP", "TCPIPDoSTCP"],
# # # #     "MQTT": ["MQTTDDoSConnectFlood", "MQTTDoSPublishFlood", "MQTTDDoSPublishFlood", "MQTTDoSConnectFlood","MQTTMalformedData"],
# # # #     "SPOOFING": ["ARPSpoofing"],
# # # #     "RECON": ["ReconPortScan", "ReconOSScan", "ReconVulScan", "ReconPingSweep"],
# # # #     "Benign": ["Benign"]
# # # # }
# # # #
# # # # # Mapping labels to new categories for 2 classes
# # # # label_map_2_classes = {
# # # #     "Benign": ["Benign"],
# # # #     "Attack": ["TCPIPDDoSUDP", "TCPIPDDoSICMP", "TCPIPDDoSTCP", "TCPIPDDoSSYN", "TCPIPDoSUDP", "TCPIPDoSSYN",
# # # #                "TCPIPDoSICMP", "TCPIPDoSTCP", "MQTTDDoSConnectFlood", "ReconPortScan", "MQTTDoSPublishFlood",
# # # #                "MQTTDDoSPublishFlood", "ReconOSScan", "ARPSpoofing", "MQTTDoSConnectFlood", "MQTTMalformedData",
# # # #                "ReconVulScan", "ReconPingSweep"]
# # # # }
# # # #
# # # # # Function to map labels to new categories
# # # # def map_labels_to_categories(labels, label_map):
# # # #     categories = []
# # # #     for label in labels:
# # # #         for category, labels_list in label_map.items():
# # # #             if label in labels_list:
# # # #                 categories.append(category)
# # # #                 break
# # # #     return categories
# # # #
# # # # # Map labels to categories for 6 classes
# # # # labels_6_class = map_labels_to_categories(train_df['label'], label_map_6_classes)
# # # # labels_6_class_test = map_labels_to_categories(test_df['label'], label_map_6_classes)
# # # #
# # # # # Map labels to categories for 2 classes
# # # # labels_2_class = map_labels_to_categories(train_df['label'], label_map_2_classes)
# # # # labels_2_class_test = map_labels_to_categories(test_df['label'], label_map_2_classes)
# # # #
# # # # #Label count
# # # # labels_6_class = pd.Series(labels_6_class)
# # # # labels_2_class = pd.Series(labels_2_class)
# # # # labels_6_class_test = pd.Series(labels_6_class_test)
# # # # labels_2_class_test = pd.Series(labels_2_class_test)
# # # # print(labels_2_class.shape)
# # # # print(labels_6_class.shape)
# # # # print(train_df.shape)
# # # # print(labels_2_class_test.shape)
# # # # print(labels_6_class_test.shape)
# # # #
# # # # label_counts_6_classes = labels_6_class.value_counts()
# # # # print("6 Classes Value Count\n",label_counts_6_classes)
# # # # label_counts_2_classes = labels_2_class.value_counts()
# # # # print("2 Classes Value Count\n",label_counts_2_classes)
# # #
# # # ## --------------------------------------------------------------------------------------------
# # # ## older code with acc 0.80
# # # #
# # # # import os
# # # # import pandas as pd
# # # # from keras.models import Sequential
# # # # from keras.layers import LSTM, Dropout, Dense
# # # # from sklearn.model_selection import train_test_split
# # # # from sklearn.preprocessing import MinMaxScaler, LabelEncoder
# # # # import matplotlib.pyplot as plt
# # # #
# # # # # Function to merge files in a folder and label them
# # # # def merge_files_in_folder(folder_path):
# # # #     # Initialize an empty list to store DataFrames
# # # #     dfs = []
# # # #
# # # #     # Loop through each file in the folder
# # # #     for filename in os.listdir(folder_path):
# # # #         # Extract label from filename and remove numbers at the end
# # # #         label = filename.split("_train.pcap.csv")[0].split("_test.pcap.csv")[0].split("1234567890")[0]
# # # #         # Remove numeric part if it exists
# # # #         label = ''.join(filter(str.isalpha, label))
# # # #         # Read the file into a DataFrame
# # # #         filepath = os.path.join(folder_path, filename)
# # # #         df = pd.read_csv(filepath)
# # # #
# # # #         # Add a 'label' column with the current label
# # # #         df['label'] = label
# # # #
# # # #         # Append the DataFrame to the list
# # # #         dfs.append(df)
# # # #
# # # #     # Concatenate all DataFrames in the list
# # # #     merged_data = pd.concat(dfs, ignore_index=True)
# # # #
# # # #     # Display value counts of labels
# # # #     print("Value counts of labels in train data:")
# # # #     print(merged_data['label'].value_counts())
# # # #     return merged_data
# # # #
# # # # # Define the paths to the training and test datasets
# # # # train_folder_path = r"C:\Users\20238023\Downloads\network\network\train dataset"
# # # # test_folder_path = r"C:\Users\20238023\Downloads\network\network\test dataset"
# # # #
# # # # # Merge files in the train and test folders
# # # # train_df = merge_files_in_folder(train_folder_path)
# # # # test_df = merge_files_in_folder(test_folder_path)
# # # #
# # # # # Assuming 'label' is the target variable
# # # # X_train = train_df.drop(columns=['label'])
# # # # y_train = train_df['label']
# # # # X_test = test_df.drop(columns=['label'])
# # # # y_test = test_df['label']
# # # #
# # # # # Encode the labels if they are categorical
# # # # label_encoder = LabelEncoder()
# # # # y_train = label_encoder.fit_transform(y_train)
# # # # y_test = label_encoder.transform(y_test)
# # # #
# # # # # Normalize the features
# # # # scaler = MinMaxScaler()
# # # # X_train_scaled = scaler.fit_transform(X_train)
# # # # X_test_scaled = scaler.transform(X_test)
# # # #
# # # # # Reshape data for LSTM (samples, timesteps, features)
# # # # X_train_scaled = X_train_scaled.reshape(X_train_scaled.shape[0], 1, X_train_scaled.shape[1])
# # # # X_test_scaled = X_test_scaled.reshape(X_test_scaled.shape[0], 1, X_test_scaled.shape[1])
# # # #
# # # # # Step 2: Define the LSTM Model
# # # # model = Sequential([
# # # #     LSTM(128, input_shape=(X_train_scaled.shape[1], X_train_scaled.shape[2]), return_sequences=True),
# # # #     Dropout(0.2),
# # # #     LSTM(64, return_sequences=False),
# # # #     Dropout(0.2),
# # # #     Dense(32, activation='relu'),
# # # #     Dense(1, activation='sigmoid')  # 'sigmoid' for binary classification, 'softmax' for multi-class
# # # # ])
# # # #
# # # # # Step 3: Compile the Model
# # # # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# # # #
# # # # # Step 4: Train the Model (with validation data)
# # # # history = model.fit(X_train_scaled, y_train, epochs=20, batch_size=64, validation_data=(X_test_scaled, y_test))
# # # #
# # # # # Step 5: Evaluate the Model (on test data)
# # # # loss, accuracy = model.evaluate(X_test_scaled, y_test)
# # # # print(f"Test Loss: {loss}")
# # # # print(f"Test Accuracy: {accuracy}")
# # # #
# # # # # Optional: Save the trained model
# # # # model.save('lstm_iomt_model.h5')
# # # #
# # # # # Step 6: Visualize the Training and Validation Accuracy
# # # # plt.plot(history.history['accuracy'], label='Train Accuracy')
# # # # plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
# # # # plt.legend()
# # # # plt.title('Model Accuracy over Epochs')
# # # # plt.xlabel('Epochs')
# # # # plt.ylabel('Accuracy')
# # # # plt.show()
# # # #
# # # # # Optional: Visualize Training and Validation Loss
# # # # plt.plot(history.history['loss'], label='Train Loss')
# # # # plt.plot(history.history['val_loss'], label='Validation Loss')
# # # # plt.legend()
# # # # plt.title('Model Loss over Epochs')
# # # # plt.xlabel('Epochs')
# # # # plt.ylabel('Loss')
# # # # plt.show()
# # # #
# # # # # Mapping labels to new categories for 6 classes
# # # # label_map_6_classes = {
# # # #     "DDos": ["TCPIPDDoSUDP", "TCPIPDDoSICMP", "TCPIPDDoSTCP", "TCPIPDDoSSYN"],
# # # #     "Dos": ["TCPIPDoSUDP", "TCPIPDoSSYN", "TCPIPDoSICMP", "TCPIPDoSTCP"],
# # # #     "MQTT": ["MQTTDDoSConnectFlood", "MQTTDoSPublishFlood", "MQTTDDoSPublishFlood", "MQTTDoSConnectFlood", "MQTTMalformedData"],
# # # #     "SPOOFING": ["ARPSpoofing"],
# # # #     "RECON": ["ReconPortScan", "ReconOSScan", "ReconVulScan", "ReconPingSweep"],
# # # #     "Benign": ["Benign"]
# # # # }
# # # #
# # # # # Mapping labels to new categories for 2 classes
# # # # label_map_2_classes = {
# # # #     "Benign": ["Benign"],
# # # #     "Attack": ["TCPIPDDoSUDP", "TCPIPDDoSICMP", "TCPIPDDoSTCP", "TCPIPDDoSSYN", "TCPIPDoSUDP", "TCPIPDoSSYN",
# # # #                "TCPIPDoSICMP", "TCPIPDoSTCP", "MQTTDDoSConnectFlood", "ReconPortScan", "MQTTDoSPublishFlood",
# # # #                "MQTTDDoSPublishFlood", "ReconOSScan", "ARPSpoofing", "MQTTDoSConnectFlood", "MQTTMalformedData",
# # # #                "ReconVulScan", "ReconPingSweep"]
# # # # }
# # # #
# # # # # Function to map labels to new categories
# # # # def map_labels_to_categories(labels, label_map):
# # # #     categories = []
# # # #     for label in labels:
# # # #         for category, labels_list in label_map.items():
# # # #             if label in labels_list:
# # # #                 categories.append(category)
# # # #                 break
# # # #     return categories
# # # #
# # # # # Map labels to categories for 6 classes
# # # # labels_6_class = map_labels_to_categories(train_df['label'], label_map_6_classes)
# # # # labels_6_class_test = map_labels_to_categories(test_df['label'], label_map_6_classes)
# # # #
# # # # # Map labels to categories for 2 classes
# # # # labels_2_class = map_labels_to_categories(train_df['label'], label_map_2_classes)
# # # # labels_2_class_test = map_labels_to_categories(test_df['label'], label_map_2_classes)
# # # #
# # # # # Label count
# # # # labels_6_class = pd.Series(labels_6_class)
# # # # labels_2_class = pd.Series(labels_2_class)
# # # # labels_6_class_test = pd.Series(labels_6_class_test)
# # # # labels_2_class_test = pd.Series(labels_2_class_test)
# # # #
# # # # print(labels_2_class.shape)
# # # # print(labels_6_class.shape)
# # # # print(train_df.shape)
# # # # print(labels_2_class_test.shape)
# # # # print(labels_6_class_test.shape)
# # # #
# # # # label_counts_6_classes = labels_6_class.value_counts()
# # # # print("6 Classes Value Count\n", label_counts_6_classes)
# # # # label_counts_2_classes = labels_2_class.value_counts()
# # # # print("2 Classes Value Count\n", label_counts_2_classes)
# # #
# # #
# # # ## ------------------------------------ end of older code
# # #
# # # import os
# # # import pandas as pd
# # # from keras.models import Sequential
# # # from keras.layers import LSTM, Dropout, Dense
# # # from sklearn.model_selection import train_test_split
# # # from sklearn.preprocessing import MinMaxScaler, LabelEncoder
# # # from keras.callbacks import EarlyStopping, ModelCheckpoint
# # # import matplotlib.pyplot as plt
# # #
# # # # Function to merge files in a folder and label them
# # # def merge_files_in_folder(folder_path):
# # #     dfs = []
# # #     for filename in os.listdir(folder_path):
# # #         label = filename.split("_train.pcap.csv")[0].split("_test.pcap.csv")[0].split("1234567890")[0]
# # #         label = ''.join(filter(str.isalpha, label))
# # #         filepath = os.path.join(folder_path, filename)
# # #         df = pd.read_csv(filepath)
# # #         df['label'] = label
# # #         dfs.append(df)
# # #     merged_data = pd.concat(dfs, ignore_index=True)
# # #     print("Value counts of labels in train data:")
# # #     print(merged_data['label'].value_counts())
# # #     return merged_data
# # #
# # # # Define paths to the datasets
# # # train_folder_path = r"C:\Users\20238023\Downloads\network\network\train dataset"
# # # test_folder_path = r"C:\Users\20238023\Downloads\network\network\test dataset"
# # #
# # # train_df = merge_files_in_folder(train_folder_path)
# # # test_df = merge_files_in_folder(test_folder_path)
# # #
# # # # Assuming 'label' is the target variable
# # # X_train = train_df.drop(columns=['label'])
# # # y_train = train_df['label']
# # # X_test = test_df.drop(columns=['label'])
# # # y_test = test_df['label']
# # #
# # # # Encode the labels (multi-class classification)
# # # label_encoder = LabelEncoder()
# # # y_train = label_encoder.fit_transform(y_train)
# # # y_test = label_encoder.transform(y_test)
# # #
# # # # Normalize the features
# # # scaler = MinMaxScaler()
# # # X_train_scaled = scaler.fit_transform(X_train)
# # # X_test_scaled = scaler.transform(X_test)
# # #
# # # # Reshape for LSTM
# # # X_train_scaled = X_train_scaled.reshape(X_train_scaled.shape[0], 1, X_train_scaled.shape[1])
# # # X_test_scaled = X_test_scaled.reshape(X_test_scaled.shape[0], 1, X_test_scaled.shape[1])
# # #
# # # # Update model for multi-class classification (6 classes or 2 classes)
# # # num_classes = len(label_encoder.classes_)
# # #
# # # # Step 2: Define the LSTM Model
# # # model = Sequential([
# # #     LSTM(128, input_shape=(X_train_scaled.shape[1], X_train_scaled.shape[2]), return_sequences=True),
# # #     Dropout(0.2),
# # #     LSTM(64, return_sequences=False),
# # #     Dropout(0.2),
# # #     Dense(32, activation='relu'),
# # #     Dense(num_classes, activation='softmax')  # Softmax for multi-class classification
# # # ])
# # #
# # # # Step 3: Compile the Model (use categorical crossentropy for multi-class)
# # # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# # #
# # # # Step 4: EarlyStopping and ModelCheckpoint callbacks
# # # early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
# # # checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)
# # #
# # # # Step 5: Train the Model with validation data
# # # history = model.fit(X_train_scaled, pd.get_dummies(y_train), epochs=50, batch_size=64,
# # #                     validation_data=(X_test_scaled, pd.get_dummies(y_test)),
# # #                     callbacks=[early_stopping, checkpoint])
# # #
# # # # Step 6: Evaluate the Model (on test data)
# # # loss, accuracy = model.evaluate(X_test_scaled, pd.get_dummies(y_test))
# # # print(f"Test Loss: {loss}")
# # # print(f"Test Accuracy: {accuracy}")
# # #
# # # # Optional: Save the trained model
# # # model.save('lstm_iomt_model.h5')
# # #
# # # # Step 7: Visualize the Training and Validation Accuracy
# # # plt.plot(history.history['accuracy'], label='Train Accuracy')
# # # plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
# # # plt.legend()
# # # plt.title('Model Accuracy over Epochs')
# # # plt.xlabel('Epochs')
# # # plt.ylabel('Accuracy')
# # # plt.show()
# # #
# # # # Optional: Visualize Training and Validation Loss
# # # plt.plot(history.history['loss'], label='Train Loss')
# # # plt.plot(history.history['val_loss'], label='Validation Loss')
# # # plt.legend()
# # # plt.title('Model Loss over Epochs')
# # # plt.xlabel('Epochs')
# # # plt.ylabel('Loss')
# # # plt.show()
# #
# #
# # import os
# # import pandas as pd
# # import numpy as np
# # import tensorflow as tf
# # from sklearn.model_selection import train_test_split
# # from sklearn.preprocessing import StandardScaler
# # from tensorflow.keras import layers, models
# # from tensorflow.keras.callbacks import EarlyStopping
# # import matplotlib.pyplot as plt
# # import seaborn as sns
# # from sklearn.metrics import confusion_matrix, classification_report
# #
# # # Function to merge files in folder
# # def merge_files_in_folder(folder_path):
# #     dfs = []
# #     for filename in os.listdir(folder_path):
# #         label = filename.split("_train.pcap.csv")[0].split("_test.pcap.csv")[0].split("1234567890")[0]
# #         label = ''.join(filter(str.isalpha, label))
# #         filepath = os.path.join(folder_path, filename)
# #         df = pd.read_csv(filepath)
# #         df['label'] = label
# #         dfs.append(df)
# #     merged_data = pd.concat(dfs, ignore_index=True)
# #     print("Value counts of labels in train data:")
# #     print(merged_data['label'].value_counts())
# #     return merged_data
# #
# # # Merge train and test datasets
# # train_folder_path = r"C:\Users\20238023\Downloads\network\network\train dataset"
# # test_folder_path = r"C:\Users\20238023\Downloads\network\network\test dataset"
# # train_df = merge_files_in_folder(train_folder_path)
# # test_df = merge_files_in_folder(test_folder_path)
# #
# # # Preprocessing data
# # # Separate features and labels
# # X_train = train_df.drop(columns=['label'])
# # y_train = train_df['label']
# # X_test = test_df.drop(columns=['label'])
# # y_test = test_df['label']
# #
# # # Encode labels if they are categorical
# # from sklearn.preprocessing import LabelEncoder
# # encoder = LabelEncoder()
# # y_train_encoded = encoder.fit_transform(y_train)
# # y_test_encoded = encoder.transform(y_test)
# #
# # # Normalize the data
# # scaler = StandardScaler()
# # X_train_scaled = scaler.fit_transform(X_train)
# # X_test_scaled = scaler.transform(X_test)
# #
# # # Train-validation split (use a portion of the training data for validation)
# # X_train_scaled, X_val_scaled, y_train_encoded, y_val_encoded = train_test_split(
# #     X_train_scaled, y_train_encoded, test_size=0.2, random_state=42)
# #
# # # Build the model
# # model = models.Sequential()
# #
# # # Input layer
# # model.add(layers.InputLayer(input_shape=(X_train_scaled.shape[1],)))
# #
# # # Hidden layers
# # model.add(layers.Dense(128, activation='relu'))
# # model.add(layers.Dropout(0.5))
# # model.add(layers.Dense(64, activation='relu'))
# # model.add(layers.Dropout(0.5))
# #
# # # Output layer
# # num_classes = len(encoder.classes_)  # Number of unique labels
# # model.add(layers.Dense(num_classes, activation='softmax'))  # Use softmax for multi-class classification
# #
# # # Compile the model with the categorical crossentropy loss for multi-class classification
# # model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
# #               loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# #
# # # Early stopping to prevent overfitting
# # early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
# #
# # # Train the model
# # history = model.fit(X_train_scaled, y_train_encoded, epochs=20, batch_size=64,
# #                     validation_data=(X_val_scaled, y_val_encoded),
# #                     callbacks=[early_stopping])
# #
# # # Evaluate the model on the test data
# # test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test_encoded)
# # print(f"Test Loss: {test_loss}")
# # print(f"Test Accuracy: {test_accuracy}")
# #
# # # Predict on test data
# # y_pred = model.predict(X_test_scaled)
# # y_pred_classes = np.argmax(y_pred, axis=1)  # Get the class with the highest probability
# #
# # # Classification Report
# # print("\nClassification Report:")
# # print(classification_report(y_test_encoded, y_pred_classes, target_names=encoder.classes_))
# #
# # # Confusion Matrix
# # cm = confusion_matrix(y_test_encoded, y_pred_classes)
# #
# # # Plot the Confusion Matrix
# # plt.figure(figsize=(10, 8))
# # sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=encoder.classes_, yticklabels=encoder.classes_)
# # plt.xlabel('Predicted')
# # plt.ylabel('True')
# # plt.title('Confusion Matrix')
# # plt.show()
# #
# # # Plot training history (Loss & Accuracy)
# # plt.figure(figsize=(12, 4))
# #
# # # Plot loss
# # plt.subplot(1, 2, 1)
# # plt.plot(history.history['loss'], label='Train Loss')
# # plt.plot(history.history['val_loss'], label='Validation Loss')
# # plt.title('Model Loss')
# # plt.xlabel('Epochs')
# # plt.ylabel('Loss')
# # plt.legend()
# #
# # # Plot accuracy
# # plt.subplot(1, 2, 2)
# # plt.plot(history.history['accuracy'], label='Train Accuracy')
# # plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
# # plt.title('Model Accuracy')
# # plt.xlabel('Epochs')
# # plt.ylabel('Accuracy')
# # plt.legend()
# #
# # plt.show()
#
# import os
# import pandas as pd
# import numpy as np
# import tensorflow as tf
# from sklearn.model_selection import StratifiedKFold
# from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, roc_curve, auc
# from sklearn.utils.class_weight import compute_class_weight
# from imblearn.over_sampling import SMOTE
# import matplotlib.pyplot as plt
# import seaborn as sns
# from tensorflow.keras import layers, models
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.callbacks import EarlyStopping
#
#
# # Function to merge files in a folder
# def merge_files_in_folder(folder_path):
#     dfs = []
#     for filename in os.listdir(folder_path):
#         label = filename.split("_train.pcap.csv")[0].split("_test.pcap.csv")[0].split("1234567890")[0]
#         label = ''.join(filter(str.isalpha, label))
#         filepath = os.path.join(folder_path, filename)
#         df = pd.read_csv(filepath)
#         df['label'] = label
#         dfs.append(df)
#     merged_data = pd.concat(dfs, ignore_index=True)
#     print("Value counts of labels in the merged data:")
#     print(merged_data['label'].value_counts())
#     return merged_data
#
#
# # Merge train and test datasets
# train_folder_path = r"C:\Users\20238023\Downloads\network\network\train dataset"
# test_folder_path = r"C:\Users\20238023\Downloads\network\network\test dataset"
# train_df = merge_files_in_folder(train_folder_path)
# test_df = merge_files_in_folder(test_folder_path)
#
# # Preprocess data
# X_train = train_df.drop('label', axis=1).values
# y_train = train_df['label'].values
# X_test = test_df.drop('label', axis=1).values
# y_test = test_df['label'].values
#
# # Convert labels to categorical format for classification
# y_train = to_categorical(pd.factorize(y_train)[0])
# y_test = to_categorical(pd.factorize(y_test)[0])
#
# # SMOTE: Over-sample minority class in training data
# smote = SMOTE(sampling_strategy='auto', random_state=42)
# X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
#
#
# # Define the model architecture
# def create_model(input_shape, num_classes):
#     model = models.Sequential([
#         layers.Dense(128, activation='relu', input_shape=input_shape),
#         layers.Dropout(0.5),
#         layers.Dense(64, activation='relu'),
#         layers.Dropout(0.5),
#         layers.Dense(32, activation='relu'),
#         layers.Dense(num_classes, activation='softmax')
#     ])
#     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#     return model
#
#
# # Stratified k-fold cross-validation
# kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#
# # Store metrics for each fold
# fold_accuracies = []
# fold_f1_scores = []
#
# for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_resampled, np.argmax(y_train_resampled, axis=1))):
#     print(f"Training fold {fold + 1}...")
#
#     # Split data
#     X_train_fold, X_val_fold = X_train_resampled[train_idx], X_train_resampled[val_idx]
#     y_train_fold, y_val_fold = y_train_resampled[train_idx], y_train_resampled[val_idx]
#
#     # Compute class weights to handle class imbalance
#     class_weights = compute_class_weight('balanced', classes=np.unique(np.argmax(y_train_fold, axis=1)),
#                                          y=np.argmax(y_train_fold, axis=1))
#     class_weights_dict = dict(enumerate(class_weights))
#
#     # Create and train model
#     model = create_model(X_train_fold.shape[1:], y_train_fold.shape[1])
#
#     early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
#     model.fit(X_train_fold, y_train_fold, epochs=20, batch_size=64, validation_data=(X_val_fold, y_val_fold),
#               class_weight=class_weights_dict, callbacks=[early_stopping])
#
#     # Evaluate on validation data
#     val_loss, val_accuracy = model.evaluate(X_val_fold, y_val_fold, verbose=0)
#     fold_accuracies.append(val_accuracy)
#
#     # Predict on validation set
#     y_pred = model.predict(X_val_fold)
#     y_pred_classes = np.argmax(y_pred, axis=1)
#     y_true = np.argmax(y_val_fold, axis=1)
#
#     # Compute F1-score
#     from sklearn.metrics import f1_score
#
#     f1 = f1_score(y_true, y_pred_classes, average='macro')
#     fold_f1_scores.append(f1)
#
# # Final Evaluation on test set
# test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
# y_pred_test = model.predict(X_test)
# y_pred_classes_test = np.argmax(y_pred_test, axis=1)
# y_true_test = np.argmax(y_test, axis=1)
#
# print(f"Test Accuracy: {test_accuracy:.4f}")
#
# # Classification report and confusion matrix
# print("\nClassification Report:\n", classification_report(y_true_test, y_pred_classes_test))
# conf_matrix = confusion_matrix(y_true_test, y_pred_classes_test)
#
# # Plot confusion matrix
# plt.figure(figsize=(10, 8))
# sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_true_test),
#             yticklabels=np.unique(y_true_test))
# plt.xlabel('Predicted')
# plt.ylabel('True')
# plt.title('Confusion Matrix')
# plt.show()
#
# # Precision-Recall curve
# precision, recall, _ = precision_recall_curve(y_true_test, y_pred_test[:, 1])
# plt.figure(figsize=(8, 6))
# plt.plot(recall, precision, label='Precision-Recall curve', color='blue')
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.title('Precision-Recall Curve')
# plt.show()
#
# # ROC curve
# fpr, tpr, _ = roc_curve(y_true_test, y_pred_test[:, 1])
# roc_auc = auc(fpr, tpr)
# plt.figure(figsize=(8, 6))
# plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})', color='green')
# plt.plot([0, 1], [0, 1], linestyle='--', color='red')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curve')
# plt.legend(loc='lower right')
# plt.show()
#
# # Final Evaluation Metrics
# print("\nAverage Accuracy across all folds:", np.mean(fold_accuracies))
# print("Average F1 Score across all folds:", np.mean(fold_f1_scores))
#




import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Function to merge files in folder
def merge_files_in_folder(folder_path):
    dfs = []
    for filename in os.listdir(folder_path):
        label = filename.split("_train.pcap.csv")[0].split("_test.pcap.csv")[0].split("1234567890")[0]
        label = ''.join(filter(str.isalpha, label))
        filepath = os.path.join(folder_path, filename)
        df = pd.read_csv(filepath)
        df['label'] = label
        dfs.append(df)
    merged_data = pd.concat(dfs, ignore_index=True)
    print("Value counts of labels in train data:")
    print(merged_data['label'].value_counts())
    return merged_data

# Merge train and test datasets
train_folder_path = r"C:\Users\20238023\Downloads\network\network\train dataset"
test_folder_path = r"C:\Users\20238023\Downloads\network\network\test dataset"
train_df = merge_files_in_folder(train_folder_path)
test_df = merge_files_in_folder(test_folder_path)

# Preprocessing data
# Separate features and labels
X_train = train_df.drop(columns=['label'])
y_train = train_df['label']
X_test = test_df.drop(columns=['label'])
y_test = test_df['label']

# Encode labels if they are categorical
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
y_train_encoded = encoder.fit_transform(y_train)
y_test_encoded = encoder.transform(y_test)

# Normalize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train-validation split (use a portion of the training data for validation)
X_train_scaled, X_val_scaled, y_train_encoded, y_val_encoded = train_test_split(
    X_train_scaled, y_train_encoded, test_size=0.2, random_state=42)

# Build the model
model = models.Sequential()

# Input layer
model.add(layers.InputLayer(input_shape=(X_train_scaled.shape[1],)))

# Hidden layers
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.5))

# Output layer
num_classes = len(encoder.classes_)  # Number of unique labels
model.add(layers.Dense(num_classes, activation='softmax'))  # Use softmax for multi-class classification

# Compile the model with the categorical crossentropy loss for multi-class classification
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
history = model.fit(X_train_scaled, y_train_encoded, epochs=20, batch_size=64,
                    validation_data=(X_val_scaled, y_val_encoded),
                    callbacks=[early_stopping])

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test_encoded)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# Predict on test data
y_pred = model.predict(X_test_scaled)
y_pred_classes = np.argmax(y_pred, axis=1)  # Get the class with the highest probability

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test_encoded, y_pred_classes, target_names=encoder.classes_))

# Confusion Matrix
cm = confusion_matrix(y_test_encoded, y_pred_classes)

# Plot the Confusion Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=encoder.classes_, yticklabels=encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Plot training history (Loss & Accuracy)
plt.figure(figsize=(12, 4))

# Plot loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
