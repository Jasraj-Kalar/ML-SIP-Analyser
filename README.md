# SIP Log Anomaly & Event Classifier

This project was undertaken as a learning exercise to explore the application of Machine Learning and Natural Language Processing (NLP) to Session Initiation Protocol (SIP) logs. The primary goal was to build a model capable of classifying different types of SIP events (e.g., normal calls, call failures) and to investigate its potential for anomaly detection.

# Findings & Conclusion

The model achieves a high accuracy on the dataset. However, this along with additional analysis suggests that the classification of SIP events is largely deterministic. The model is effectively re-learning rules that are already explicit in the SIP protocol (e.g., a 404 Not Found response indicates a failed call).

While this serves as a successful proof-of-concept for applying NLP, the results suggest that for the classification of SIP logs, a traditional rule-based system would likely be more efficient. Such a system would directly parse the standardised SIP response codes/flows rather than inferring the outcome through a probabilistic model.

The model's "anomaly score" (calculated as 1 - prediction_confidence) is effective at flagging samples that the classifier finds ambiguous or difficult to categorise. This is useful for identifying unusual variations within the known classes. However, the methodology would need to be changed to support the detection of novel anomalies.
