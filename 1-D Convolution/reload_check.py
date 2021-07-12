from src import *

dir = 'C:/Users/mask2/Desktop/In-Class/DNN PROJECT/data/genres/'
genres = {'blues': 0, 'classical': 1, 'country': 2, 'disco': 3, 'hiphop': 4,'jazz': 5, 'pop': 6, 'metal': 7, 'reggae': 8, 'rock': 9}
SR=44100

version=2.0
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


version=2.0
Model_NAME='CUSTOM_1D_NETWORK_'+str(version)

model_1 = load_model(Model_NAME + ".h5")
history=np.load(Model_NAME + "history.npy",allow_pickle='TRUE').item()

print("")
for key,values in history.items():
    print(key,":",values)



test_pred = model_1.predict_classes(test_set)
test_proba = model_1.predict_proba(test_set)

print("Accuracy:",accuracy_score(test_classes, test_pred))

fig1=skplt.metrics.plot_confusion_matrix(test_classes, test_pred, normalize=True)
#fig1.savefig('Confusion_Matrix_2.0.png',dpi=300)

fig2=plt.figure()
# Plot training & validation accuracy values
plt.plot(history['accuracy'])
plt.plot(history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('Accuracy_Model_2.0.png',dpi=300)

fig3=plt.figure()
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('Loss_Model_2.0.png',dpi=300)
plt.show()
