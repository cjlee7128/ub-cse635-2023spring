import json 

from flair.nn import Classifier 
from flair.data import Sentence 

# load the model
tagger = Classifier.load('upos') 

def convertToPOS(sentence): 
    _sentence = Sentence(sentence) 
    # predict POS tags 
    tagger.predict(_sentence) 
    return ' '.join([token.text + '/' + token.tag for token in _sentence.tokens]) 

def createPOSData(data_name): 
    with open(f'data/{data_name}.json', 'r') as f: 
        data = json.load(f) 

    output_data = [] 
    for conversation in data: 
        for turn in conversation['utterances']: 

            history = turn['history'][-1] 
            output_data.append({"sentence": history, "POS": convertToPOS(history)}) 

            original_response = turn['original_response'] 
            if original_response is not None: 
                output_data.append({"sentence": original_response, "POS": convertToPOS(original_response)}) 

            knowledge = turn["knowledge"] 
            output_data.append({"sentence": knowledge, "POS": convertToPOS(knowledge)}) 

            response = turn["response"] 
            output_data.append({"sentence": response, "POS": convertToPOS(response)}) 

    with open(f'data/{data_name}_pos.json', 'w') as outfile: 
        json_object = json.dumps(output_data, indent=2) 
        outfile.write(json_object) 

createPOSData('train') 
print('train done') 
createPOSData('valid') 
print('valid done') 
createPOSData('test') 
print('test done') 