import json
import fastapi
import modules.ner as ner
from fastapi import FastAPI
from pydantic import BaseModel

text = """Mohammed Salisu hatte den Starangreifer im Strafraum am Fuß getroffen, aber auch den Ball gespielt. Es gab dennoch Strafstoß, und diese Chance ließ sich Ronaldo nicht entgehen (65.). Ronaldo ist nun der erste Spieler, der bei fünf Endrunden getroffen hat.
    Doch allzu lang konnte sich Portugal nicht über den historischen Treffer freuen. Erst scheiterte Kudus zwar noch mit einem weiteren Distanzschuss, dann legte der Angreifer jedoch für André Ayew auf, der aus kurzer Distanz ins Tor schob (73.). Kurios: Beide wurden kurz darauf ausgewechselt.
    Wie würde Portugal mit diesem Rückschlag umgehen? Mit einem Doppelpack. Zwei Vorlagen von Bruno Fernandes verwandelten zunächst João Felix mit einem Heber (78.) und der eingewechselte Rafael Leão mit seinem ersten Ballkontakt (80.)"""

sentences = ner.tag(text)

class Sents:
    entities = []
    pos = []

class SentsEncoder(json.JSONEncoder):
    def default(self, obj):
            return {'ents': obj.entities, 'pos': obj.pos}

sents = Sents()

# iterate through sentences and print predicted labels
for sentence in sentences:
    upos_multi_fast = json.dumps(sentence.to_dict(tag_type='flair/upos-multi-fast'))
    ner_multi_fast = json.dumps(sentence.to_dict(tag_type='flair/ner-multi-fast'))
    print(sentence)
    print('--------------')
    print(sentence.to_tagged_string())
    print('--------------')
    print(dir(sentence))
    # Result:
    # ['_Sentence__remove_zero_width_characters', '_Sentence__restore_windows_1252_characters', '__class__', '__delattr__', '__dict__', '__dir__', '__doc__',
    # '__eq__', '__format__', '__ge__', '__getattribute__', '__getitem__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__iter__', '__le__', '__len__',
    # '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__',
    # '_embeddings', '_handle_problem_characters', '_known_spans', '_metadata', '_next_sentence', '_position_in_dataset', '_previous_sentence', '_printout_labels',
    # 'add_label', 'add_metadata', 'add_token', 'annotation_layers', 'clear_embeddings', 'embedding', 'end_pos', 'end_position', 'get_each_embedding', 'get_embedding',
    # 'get_label', 'get_labels', 'get_language_code', 'get_metadata', 'get_relations', 'get_span', 'get_spans', 'get_token', 'has_label', 'has_metadata',
    # 'infer_space_after', 'is_context_set', 'is_document_boundary', 'labels', 'language_code', 'left_context', 'next_sentence', 'previous_sentence', 'remove_labels',
    # 'right_context', 'score', 'set_embedding', 'set_label', 'start_pos', 'start_position', 'tag', 'text', 'to', 'to_dict', 'to_original_text', 'to_plain_string',
    # 'to_tagged_string', 'to_tokenized_string', 'tokenized', 'tokens', 'unlabeled_identifier']
    print('--------------')
    for span in sentence.get_spans('flair/upos-multi-fast'):
        print(span)
        print("upos-multi-fast span")
        print(dir(span))
        # ['__abstractmethods__', '__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__getitem__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__iter__', '__le__', '__len__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__slots__', '__str__', '__subclasshook__', '__weakref__', '_abc_impl', '_embeddings', '_init_labels', '_metadata', '_printout_labels', 'add_label', 'add_metadata', 'annotation_layers', 'clear_embeddings', 'embedding', 'end_position', 'get_each_embedding', 'get_embedding', 'get_label', 'get_labels', 'get_metadata', 'has_label', 'has_metadata', 'labels', 'remove_labels', 'score', 'sentence', 'set_embedding', 'set_label', 'start_position', 'tag', 'text', 'to', 'tokens', 'unlabeled_identifier']
        print(f"start_position: {span.start_position}, end_position: {span.end_position}, tag: {span.tag}")
    for span in sentence.get_spans('flair/ner-multi-fast'):
        print("ner-multi-fast span")
        print(dir(span))
        print(f"start_position: {span.start_position}, end_position: {span.end_position}, tag: {span.tag}, text: {span.text}")
    print('--------------')
    print(sentence.tokens)
    for token in sentence.tokens:
        print(f"start_pos: {token.start_pos}, end_pos: {token.end_pos}, tag: {token.tag}, text: {token.text}")
    print('--------------')
    print(ner_multi_fast)
    print('--------------')
    print(upos_multi_fast)
    

class Input(BaseModel):
    text: str
    description: str | None = None

app = FastAPI()

@app.get("/tag")
async def read_item(input: Input | None = None):
    if input:
        sentences = ner.tag(input.text)
        sents = Sents()
        j = ""
        for sentence in sentences:
            j += sentence.to_tagged_string()
            for span in sentence.get_spans('flair/ner-multi-fast'):
                sents.entities.append({'start_pos ': span.start_position, 'end_pos': span.end_position, 'tag': span.tag, 'text': span.text})   
            for token in sentence.tokens:
                sents.pos.append({'start_pos ': token.start_position, 'end_pos': token.end_position, 'tag': token.tag, 'text': token.text})  

        return json.dumps(sents, cls=SentsEncoder)
    return "moin"