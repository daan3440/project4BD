from pyspark import SparkContext
from pyspark.sql import SQLContext, DataFrame
from pyspark.ml import Pipeline
from pyspark.ml.feature import IDF, Tokenizer, CountVectorizer, StopWordsRemover, StringIndexer
from pyspark.ml.classification import DecisionTreeClassifier, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import sys
from functools import reduce

#daan test
# Function for combining multiple DataFrames row-wise
def unionAll(*dfs):
    return reduce(DataFrame.unionAll, dfs)

if __name__ == "__main__":
    # Create a SparkContext and an SQLContext
    sc = SparkContext(appName="Sentiment Classification")
    sqlContext = SQLContext(sc)

    # Load data
    # wholeTextFiles(path, [...]) reads a directory of text files from a filesystem
    # Each file is read as a single record and returned in a key-value pair
    # The key is the path and the value is the content of each file
    reviews = sc.wholeTextFiles('hdfs://quickstart.cloudera:8020/user/cloudera/'+sys.argv[1]+'/*/*')

    # Create tuples: (class label, review text) - we ignore the file path
    # 1.0 for positive reviews
    # 0.0 for negative reviews
    reviews_f = reviews.map(lambda row: (1.0 if 'pos' in row[0] else 0.0, row[1]))

    # Convert data into a Spark SQL DataFrame
    # The first column contains the class label
    # The second column contains the review text
    dataset = reviews_f.toDF(['class_label', 'review'])

    # ----- PART II: FEATURE ENGINEERING -----

    # Tokenize the review text column into a list of words
    tokenizer = Tokenizer(inputCol='review', outputCol='words')
    words_data = tokenizer.transform(dataset)

    # Randomly split data into a training set, a development set and a test set
    # train = 60% of the data, dev = 20% of the data, test = 20% of the data
    # The random seed should be set to 42
    (train, dev, test) = words_data.randomSplit([.6, .2, .2], seed = 42)

    # TODO: Count the number of instances in, respectively, train, dev and test
    # Print the counts to standard output
    # [FIX ME!] Write code below

    train_count = train.count()
    dev_count = dev.count()
    test_count = test.count()
    print ("Train size: %d" % train_count)
    print ("Dev size: %d" % dev_count)
    print ("Test size: %d" % test_count)

    # TODO: Count the number of positive/negative instances in, respectively, train, dev and test
    # Print the class distribution for each to standard output
    # The class distribution should be specified as the % of positive examples
    # [FIX ME!] Write code below

    train_pos = train.filter(dev["class_label"] == 1.0).count()
    dev_pos = dev.filter(dev["class_label"] == 1.0).count()
    test_pos = test.filter(dev["class_label"] == 1.0).count()

    print("Train % pos:" + " %.2f" % (train_pos / train_count * 100))
    print("Dev % pos:" + " %.2f" % (dev_pos / dev_count * 100))
    print("Test % pos:" + " %.2f" % (test_pos / test_count * 100))



    # TODO: Create a stopword list containing the 100 most frequent tokens in the training data
    # Hint: see below for how to convert a list of (word, frequency) tuples to a list of words
    # stopwords = [frequency_tuple[0] for frequency_tuple in list_top100_tokens]
    # [FIX ME!] Write code below
    train_token_count = train.select("review").rdd.flatMap(lambda x: x[0].split()).map(lambda x: (x, 1)).reduceByKey(lambda a,b: a+b)
    top100_tokens = sorted(train_token_count.collect(), key=lambda x: x[1], reverse=True)[:100]
    stopwords = [frequency_tuple[0] for frequency_tuple in top100_tokens]

    # TODO: Replace the [] in the stopWords parameter with the name of your created list
    # [FIX ME!] Modify code below
    remover = StopWordsRemover(inputCol='words', outputCol='words_filtered', stopWords=stopwords)

    # Remove stopwords from all three subsets
    train_filtered = remover.transform(train)
    dev_filtered = remover.transform(dev)
    test_filtered = remover.transform(test)

    # Transform data to a bag of words representation
    # Only include tokens that have a minimum document frequency of 2
    cv = CountVectorizer(inputCol='words_filtered', outputCol='BoW', minDF=2.0)
    cv_model = cv.fit(train_filtered)
    train_data = cv_model.transform(train_filtered)
    dev_data = cv_model.transform(dev_filtered)
    test_data = cv_model.transform(test_filtered)

    # TODO: Print the vocabulary size (to STDOUT) after filtering out stopwords and very rare tokens
    # Hint: Look at the parameters of CountVectorizer
    # [FIX ME!] Write code below
    print("Vocab size: %d" % len(cv_model.vocabulary))

    # Create a TF-IDF representation of the data
    idf = IDF(inputCol='BoW', outputCol='TFIDF')
    idf_model = idf.fit(train_data)
    train_tfidf = idf_model.transform(train_data)
    dev_tfidf = idf_model.transform(dev_data)
    test_tfidf = idf_model.transform(test_data)

    # ----- PART III: MODEL SELECTION -----

    # Provide information about class labels: needed for model fitting
    # Only needs to be defined once for all models (but included in all pipelines)
    label_indexer = StringIndexer(inputCol = 'class_label', outputCol = 'label')

    # Create an evaluator for binary classification
    # Only needs to be created once, can be reused for all evaluation
    evaluator = BinaryClassificationEvaluator()

    # Train a decision tree with default parameters (including maxDepth=5)
    dt_classifier_default = DecisionTreeClassifier(labelCol = 'label', featuresCol = 'TFIDF', maxDepth=5)

    # Create an ML pipeline for the decision tree model
    dt_pipeline_default = Pipeline(stages=[label_indexer, dt_classifier_default])

    # Apply pipeline and train model
    dt_model_default = dt_pipeline_default.fit(train_tfidf)

    # Apply model on devlopment data
    dt_predictions_default_dev = dt_model_default.transform(dev_tfidf)

    # Evaluate model using the AUC metric
    auc_dt_default_dev = evaluator.evaluate(dt_predictions_default_dev, {evaluator.metricName: 'areaUnderROC'})

    # Print result to standard output
    print('Decision Tree, Default Parameters, Development Set, AUC: ' + str(auc_dt_default_dev))

    # TODO: Check for signs of overfitting (by evaluating the model on the training set)
    # [FIX ME!] Write code below

    # Apply model on training data
    dt_predictions_default_train = dt_model_default.transform(train_tfidf)

    # Evaluate model using the AUC metric
    auc_dt_default_train = evaluator.evaluate(dt_predictions_default_train, {evaluator.metricName: 'areaUnderROC'})

    # Print result to standard output
    print('Decision Tree, Default Parameters, Training Set, AUC: ' + str(auc_dt_default_train))

    # TODO: Tune the decision tree model by changing one of its hyperparameters
    # Build and evalute decision trees with the following maxDepth values: 3 and 4.
    # [FIX ME!] Write code below

    # Create Decision Tree with maxDepth of 3 and 4.
    dt_classifier_depth_three = DecisionTreeClassifier(labelCol = 'label', featuresCol = 'TFIDF', maxDepth=3)
    dt_classifier_depth_four = DecisionTreeClassifier(labelCol = 'label', featuresCol = 'TFIDF', maxDepth=4)

     # Create an ML pipeline for the decision tree model
    dt_pipeline_depth_three = Pipeline(stages=[label_indexer, dt_classifier_depth_three])
    dt_pipeline_depth_four = Pipeline(stages=[label_indexer, dt_classifier_depth_four])

    # Apply pipeline and train model
    dt_model_depth_three = dt_pipeline_depth_three.fit(train_tfidf)
    dt_model_depth_four = dt_pipeline_depth_four.fit(train_tfidf)

    # Apply model on development data
    dt_predictions_depth_three_dev = dt_model_depth_three.transform(dev_tfidf)
    dt_predictions_depth_four_dev = dt_model_depth_four.transform(dev_tfidf)

    # Apply model on training data
    dt_predictions_depth_three_train = dt_model_depth_three.transform(train_tfidf)
    dt_predictions_depth_four_train = dt_model_depth_four.transform(train_tfidf)

    # Evaluate model, on development data, using the AUC metric
    auc_dt_depth_three_dev = evaluator.evaluate(dt_predictions_depth_three_dev, {evaluator.metricName: 'areaUnderROC'})
    auc_dt_depth_four_dev = evaluator.evaluate(dt_predictions_depth_four_dev, {evaluator.metricName: 'areaUnderROC'})

    # Evaluate model, on training data, using the AUC metric, to check for overfitting
    auc_dt_depth_three_train = evaluator.evaluate(dt_predictions_depth_three_train, {evaluator.metricName: 'areaUnderROC'})
    auc_dt_depth_four_train = evaluator.evaluate(dt_predictions_depth_four_train, {evaluator.metricName: 'areaUnderROC'})

    # Print result to standard output, for both traing and development data
    print('Decision Tree, Depth Three, Development Set, AUC: ' + str(auc_dt_depth_three_dev))
    print('Decision Tree, Depth Four, Development Set, AUC: ' + str(auc_dt_depth_four_dev))

    print('Decision Tree, Depth Three, Training Set, AUC: ' + str(auc_dt_depth_three_train))
    print('Decision Tree, Depth Four, Training Set, AUC: ' + str(auc_dt_depth_four_train))

    # Train a random forest with default parameters (including numTrees=20)
    rf_classifier_default = RandomForestClassifier(labelCol = 'label', featuresCol = 'TFIDF', numTrees=20)

    # Create an ML pipeline for the random forest model
    rf_pipeline_default = Pipeline(stages=[label_indexer, rf_classifier_default])

    # Apply pipeline and train model
    rf_model_default = rf_pipeline_default.fit(train_tfidf)

    # Apply model on development data
    rf_predictions_default_dev = rf_model_default.transform(dev_tfidf)

    # Evaluate model using the AUC metric
    auc_rf_default_dev = evaluator.evaluate(rf_predictions_default_dev, {evaluator.metricName: 'areaUnderROC'})

    # Print result to standard output
    print('Random Forest, Default Parameters, Development Set, AUC:' + str(auc_rf_default_dev))

    # TODO: Check for signs of overfitting (by evaluating the model on the training set)
    # [FIX ME!] Write code below

    # Apply model on training data
    rf_predictions_default_train = rf_model_default.transform(train_tfidf)

    # Evaluate model using the AUC metric
    auc_rf_default_train = evaluator.evaluate(rf_predictions_default_train, {evaluator.metricName: 'areaUnderROC'})

    # Print result to standard output
    print('Random Forest, Default Parameters, Training Set, AUC:' + str(auc_rf_default_train))

    # TODO: Tune the random forest model by changing one of its hyperparameters
    # Build and evalute (on the dev set) another random forest with the following numTrees value: 100.
    # [FIX ME!] Write code below

    # Create random forest with numTrees = 100
    rf_classifier_hundred = RandomForestClassifier(labelCol = 'label', featuresCol = 'TFIDF', numTrees=100)

    # Create an ML pipeline for the random forest model
    rf_pipeline_hundred = Pipeline(stages=[label_indexer, rf_classifier_hundred])

    # Apply pipeline and train model
    rf_model_hundred = rf_pipeline_hundred.fit(train_tfidf)

    # Apply model on development data
    rf_predictions_hundred_train = rf_model_hundred.transform(train_tfidf)

    # Apply model on training data
    rf_predictions_hundred_dev = rf_model_hundred.transform(dev_tfidf)

    # Evaluate model, on development data, using the AUC metric
    auc_rf_hundred_train = evaluator.evaluate(rf_predictions_hundred_train, {evaluator.metricName: 'areaUnderROC'})

    # Evaluate model, on traing data, using the AUC metric, to check for overfitting
    auc_rf_hundred_dev = evaluator.evaluate(rf_predictions_hundred_dev, {evaluator.metricName: 'areaUnderROC'})

    # Print result to standard output for both training and development data.
    print('Random Forest, Hundred Parameters, Development Set, AUC:' + str(auc_rf_hundred_dev))
    print('Random Forest, Hundred Parameters, Training Set, AUC:' + str(auc_rf_hundred_train))

    # ----- PART IV: MODEL EVALUATION -----

    # Create a new dataset combining the train and dev sets
    traindev_tfidf = unionAll(train_tfidf, dev_tfidf)

    # TODO: Evalute the best model on the test set
    # Build a new model from the concatenation of the train and dev sets in order to better utilize the data
    # [FIX ME!]

    # Apply pipeline and train model on training/development set
    rf_model_hundred = rf_pipeline_hundred.fit(traindev_tfidf)

    # Apply model on test data
    rf_predictions_hundred_test = rf_model_hundred.transform(test_tfidf)

    # Evaluate model, on test data, using the AUC metric
    auc_rf_hundred_test = evaluator.evaluate(rf_predictions_hundred_test, {evaluator.metricName: 'areaUnderROC'})

    # Print result to standard output for test data
    print('Random Forest, Hundred Parameters, Test Set, AUC:' + str(auc_rf_hundred_test))

    # Apply model on training/development data, to check for overfitting
    rf_predictions_hundred_traindev = rf_model_hundred.transform(traindev_tfidf)

    # Evaluate model, on training/development data, using the AUC metric
    auc_rf_hundred_traindev = evaluator.evaluate(rf_predictions_hundred_traindev, {evaluator.metricName: 'areaUnderROC'})

    # Print result to standard output for training/development data
    print('Random Forest, Hundred Parameters, Training/Development Set, AUC:' + str(auc_rf_hundred_traindev))
