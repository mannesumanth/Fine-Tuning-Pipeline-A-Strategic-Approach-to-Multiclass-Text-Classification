import os
from pprint import pprint
from OpenAI import OpenAI, Prompter, Pipeline
model        = OpenAI("********************************************************",)
prompter     = Prompter('multilabel_classification.jinja')
pipe         = Pipeline(prompter , model)
labels = {'surprise', 'neutral', 'hate', 'joy', 'worry', 'sadness'}


result = pipe.fit('multiclass_classification.jinja',
                                      labels=labels,
                                      text_input="Amazing customer service.",
                                     )

pprint(eval(result['text']))