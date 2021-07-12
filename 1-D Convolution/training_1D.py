from src import *

dir = 'C:/Users/mask2/Desktop/In-Class/DNN PROJECT/data/genres/'# Change Directory to one that is required

genres = {'blues': 0, 'classical': 1, 'country': 2, 'disco': 3, 'hiphop': 4,'jazz': 5, 'pop': 6, 'metal': 7, 'reggae': 8, 'rock': 9}
SR=22050
version=3.0
Model_NAME='CUSTOM_1D_NETWORK_'+str(version)
song_data,classes=myloaddata(dir, genres)

data = np.array(song_data)
print("Input Data Shape:",data.shape)



labelencoder = LabelEncoder()
labelencoder.fit(classes)
n_classes = len(labelencoder.classes_)
print (n_classes, "classes:", ", ",(list(labelencoder.classes_)))
classes_num = labelencoder.transform(classes)
classes_num_1hot = to_categorical(classes_num)


testset_size = 0.30
splitter = StratifiedShuffleSplit(n_splits=1, test_size=testset_size, random_state=0)
splits = splitter.split(data, classes_num)
train_set,test_set,train_classes,train_classes_1hot,test_classes,test_classes_1hot = mysplitdata(splits,data,classes_num,classes_num_1hot)
print ("Training Set Shape:",train_set.shape)
print ("Test Set Shape:",test_set.shape)


input_shape = train_set.shape
model = Sequential()
model.add(Conv1D(16, 64, input_shape=input_shape[1:], activation='relu', strides=2))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=8, strides = 8))
model.add(Conv1D(32, 32, activation='relu', strides=2))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=8, strides = 8))
model.add(Conv1D(64, 16, activation='relu', strides=2))#, padding='causal'))
model.add(BatchNormalization())
model.add(Conv1D(128, 8, activation='relu', strides=2))#, padding='causal'))
model.add(BatchNormalization())
model.add(Conv1D(256, 4, activation='relu', strides=2))#, padding='causal'))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=4, strides = 4))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(n_classes,activation='softmax'))
model.compile(loss='categorical_crossentropy' , optimizer='Adadelta' , metrics=['accuracy'])
model.summary()



batch_size = 20
history = model.fit_generator(generator(train_set, train_classes_1hot, batch_size), steps_per_epoch=len(train_set),epochs=10, validation_data=generator(test_set, test_classes_1hot, batch_size), validation_steps=20)



test_pred = model.predict_classes(test_set)
test_proba = model.predict_proba(test_set)

print("Accuracy:",accuracy_score(test_classes, test_pred))
fig1=plt.figure()
skplt.metrics.plot_confusion_matrix(test_classes, test_pred, normalize=True)
plt.show()

fig2=plt.figure()
print(history.history.keys())

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()



model.save(Model_NAME + ".h5")
history_1 = history.history
#print(history_1)
for key,values in history_1.items():
    print(key,":",values)


model_1 = load_model(Model_NAME + ".h5")
np.save(Model_NAME + "history.npy", history_1)

history_check=np.load(Model_NAME + "history.npy",allow_pickle='TRUE').item()
