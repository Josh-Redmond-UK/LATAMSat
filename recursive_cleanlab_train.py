if __name__ == "__main__":
    import tensorflow as tf
    import tensorflow_hub as hub
    import tensorflow_datasets as tfds
    import glob
    import shutil
    import os
    from cleanlab import Datalab
    import pandas as pd
    import numpy as np
    import pickle
    from scriptUtils import *


        


    for i in range(5):
        folder = 'latamSatData/datasetRGB_relabel/'
        builder = tfds.folder_dataset.ImageFolder(folder)
        full_dataset = builder.as_dataset()['Train']

        def lognormalise(image, bottom_pct, top_pct):
            image = np.nan_to_num(image)
            image = np.log(image)
            min = np.percentile(image.flatten(), bottom_pct)
            max = np.percentile(image.flatten(), top_pct)
            image = (image - min)/(max-min)
            return image



        def onehot_encode(x):
            x['label'] = tf.one_hot(x['label'], 19)
            return x

        full_dataset = full_dataset.map(onehot_encode)
        full_dataset = full_dataset.batch(51200*2)


        m = make_model(input_shape=(64,64,3), num_classes=19)

        def preprocess_images(batch):
            batch['image'] = tf.cast(batch['image'], dtype=tf.float32)
            batch['image'] = batch['image'] - tf.math.reduce_mean(batch['image'])
            batch['image'] = tf.math.divide_no_nan(batch['image'], tf.math.abs(tf.math.reduce_max(batch['image'])))
            return batch
            
            



        m.compile(
            loss="categorical_crossentropy",
            optimizer="adam",
            metrics=["accuracy"],
            run_eagerly=True
        )

        base_dataset = builder.as_dataset()['Train']






        def get_xy_training(b):
            return b['image'], b['label']


        #filenames = []
        #labels = []
        base_dataset = base_dataset.map(onehot_encode)
        base_dataset = base_dataset.map(get_xy_training)
        train, test = tf.keras.utils.split_dataset(base_dataset, 0.8, 0.2, shuffle=True, seed=1)
        train = train.batch(32)
        test = test.batch(32)
        filename = f'cleanlab_model_reshuffle_iteration_{i}'
        print(f'starting iteration{i}')
        # train the model on training data for 36 epochs

        m.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
        run_eagerly=False)

        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=filename+'_checkpoint',
        save_weights_only=False,
        monitor='val_loss',
        mode='min',
        save_best_only=True)


        train_history = m.fit(train, validation_data = test, epochs=6, callbacks=[model_checkpoint_callback])
        history = train_history.history
        history_frame = pd.DataFrame([history['loss'], history['val_loss'], history['accuracy'], history['val_accuracy']],
                        index = ['Loss', 'Val_Loss', 'Accuracy', 'Val_Accuracy'])

        history_frame.to_csv(f'training_reshuffle_history_dataframe_iteration_{i}.csv')
        m = tf.keras.models.load_model(filename+'_checkpoint')
        print('model loaded')
        feature_extraction = tf.keras.Model(m.input, m.layers[-3].output)
        print('feature extraciton model created')

        issues_list = []
        labs_list = []
        filepaths = []
        j = 1
        for batch in full_dataset:
            print('batch', j)
            images = np.squeeze(np.array(batch['image'].numpy()))
            classes =np.argmax(batch['label'].numpy(), axis=1)
            paths = batch['image/filename']
            filepaths.append(paths)
            print('data loaded')
            classProb = np.squeeze(np.array(m.predict(images, verbose=0)))
            print('class probabilitys generated')
            predFeat= np.squeeze(np.array(feature_extraction.predict(images, verbose=0)))
            print('features extracted')
            all_data = {'images':images, 'classes':classes, 'probs':classProb, 'features':predFeat}
            with open('temp.pickle', 'wb') as writer:
                pickle.dump(all_data, writer)
                
            
            data_to_clean = {'Images':images, 'Labels':classes}
            print('data prepared')
            lab = Datalab(data=data_to_clean, label_name="Labels", image_key="Images")
            print('datalab created')
            _ = lab.find_issues(pred_probs=classProb, features=predFeat)
            print('isues found')
            issues = lab.get_issues('label')
            issues['filepath'] = paths
            issues_list.append(issues)
            labs_list.append(lab)
            j += 1 
            

        all_issues = pd.concat(issues_list)

        all_issues
        all_issues['given_label'] = all_issues['given_label'].map(classnameDict)
        all_issues['predicted_label'] = all_issues['predicted_label'].map(classnameDict)
        all_issues['filepath'] = all_issues['filepath'].apply(lambda x: x.decode())
        all_issues[all_issues['is_label_issue']]

        def relabel_filepath(file_path, new_label):
            file_path = file_path.split('/')
            file_path[-2]=new_label
            file_path = '/'.join(file_path)
            return file_path
        all_issues['new_filepath'] = all_issues.apply(lambda x: relabel_filepath(x.filepath, x.predicted_label), axis=1)

        def move_file(old_path, new_path):
            shutil.move(old_path, new_path)  
        all_issues.to_csv(f'cleanlab_relabel_{i}.csv')
        label_issues = all_issues[all_issues['is_label_issue']]
        print(len(label_issues), 'issues found')
        label_issues.apply(lambda x: move_file(x.filepath, x.new_filepath), axis=1)


