import os
import os.path
import pandas as pd
import numpy as np
import json
from typing import Dict, List, Tuple

import spacy
from spacy.lang import de  # for derbi
from spacy.matcher import Matcher
from spacy.attrs import *
from somajo import SoMaJo
import random 

from pattern.de import conjugate
from pattern.de import INFINITIVE, PRESENT, SG, SUBJUNCTIVE, PAST, PARTICIPLE
import pattern.text
from pattern.helpers import decode_string
from codecs import BOM_UTF8

from DERBI.derbi import DERBI

import stanza

stanza_nlp = stanza.Pipeline(lang='de', processors='tokenize,pos',
                             use_gpu=False)

resources_path = 'resources'

tokenizer = SoMaJo("de_CMC", split_camel_case=True)
nlp = spacy.load('de_core_news_sm')
derbi = DERBI(nlp)

prepositions_with_genitive = [
    line.strip() for line in
    open(f'{resources_path}/prepositions_with_genitive.txt').readlines()]
articles_dict = json.load(open(f'{resources_path}/definite_article.json'))
prepositions_dict = json.load(open(f'{resources_path}/prepositions.json'))
person_tags = ['B-' + line.strip() for line in open(
    f'{resources_path}/person_tags.txt').readlines()]
question_words = [line.strip() for line in open(
    f'{resources_path}/question_words.txt').readlines()]
auxiliary_verbs = [line.strip() for line in open(
    f'{resources_path}/auxiliary_verbs.txt').readlines()]
dawords = [line.strip() for line in open(
    f'{resources_path}/dawords.txt').readlines()]
modal_verbs = [line.strip() for line in open(
    f'{resources_path}/modal_verbs.txt').readlines()]
# Name lists via https://osf.io/jepzp/
female_names = [line.strip().lower() for line in open(
    f'{resources_path}/Names_female_Duden_2007.csv').readlines()]
male_names = [line.strip().lower() for line in open(
    f'{resources_path}/Names_male_Duden_2007.csv').readlines()]

###
# Patch generator issue in pattern:
# https://github.com/clips/pattern/issues/308#issuecomment-1308344763
BOM_UTF8 = BOM_UTF8.decode("utf-8")
decode_utf8 = decode_string


def _read(path, encoding="utf-8", comment=";;;"):
    """Returns an iterator over the lines in the file at the given path,
    strippping comments and decoding each line to Unicode.
    """
    if path:
        if isinstance(path, str) and os.path.exists(path):
            # From file path.
            f = open(path, "r", encoding="utf-8")
        elif isinstance(path, str):
            # From string.
            f = path.splitlines()
        else:
            # From file or buffer.
            f = path
        for i, line in enumerate(f):
            line = line.strip(BOM_UTF8) \
                if i == 0 and isinstance(line, str) else line
            line = line.strip()
            line = decode_utf8(line, encoding)
            if not line or (comment and line.startswith(comment)):
                continue
            yield line


pattern.text._read = _read
###


def is_ne(sentence, current_span):
    parse = nlp(sentence)
    for token in parse[current_span['start']: current_span['end']]:
        if token.ent_type_ == '':
            return False
    else:
        return True


###################### NOUN GROUPS ######################

###################### von construction instead of genitive ######################

def get_genitive_groups(sentence :str) -> List[str]:
    '''
        get spans of genitive groups 
    '''
    parse = nlp(sentence)
    genitive_groups = []
    
    for token in parse:
        if "Gen" in token.morph.get('Case')  and token.ent_type_ == '':
            genitive_group = " ".join([child.text for child in token.children])
            if genitive_group:
                genitive_groups.append(" ".join([genitive_group,token.text]))
    return genitive_groups


def genitive_group_to_dative_group(genitive_group: str) -> str:
    '''
        change a genitive group into a dative group
    '''
    info = [token.morph.to_dict() for token in nlp(genitive_group)]
    inflected_info = []
    idx = [i for i in range(len(info))]
    result = []
    for token in info:
        token['Case'] = 'Dat'
        inflected_info.append(token)

    dative_group = derbi(genitive_group, inflected_info, idx)
    for token1, token2 in zip(genitive_group.split(), dative_group.split()):
        if token1[0].isupper():
            result.append(token2[0].upper() + token2[1:])
        else:
            result.append(token2)
    
    dative_tokens = dative_group.split()
    if dative_tokens[0] == 'dem':
        dative_group = ['vom'] + result[1:]
    else:
        dative_group = ['von'] + result
    
    dative_group = " ".join(dative_group)
    return dative_group


def perturb_genitive_to_dative(
        tokens: List[str],
        tags: List[str]) -> Tuple[List[str], List[str], bool]:
    '''
        perturb sentence by changing genitive to dative
    '''
    replaced = False 
    
    for preposition in prepositions_with_genitive:
        if preposition in tokens:
            return tokens, tags, replaced
        
    sentence = ' '.join(tokens)
    perturbed_tokens = tokens.copy()
    perturbed_tags = tags.copy()
    
    genitive_groups = get_genitive_groups(sentence)
    
    if not genitive_groups:
        return tokens, tags, replaced        
    
    for genitive_group in genitive_groups:
        try:
            dative_group = genitive_group_to_dative_group(genitive_group)
            perturbed_sentence = sentence.replace(genitive_group, dative_group)
            perturbed_tokens = perturbed_sentence.split()
        except:
            pass
        
    replaced = tokens != perturbed_tokens
            

        
    for i, token in enumerate(perturbed_tokens):
        if token in ['von','vom']:
            perturbed_tags.insert(i,'O')
            
    return perturbed_tokens, perturbed_tags, replaced

###################### Possessive dative instead of genitive ######################


def capitalize_sentence(sentence):
    capitalized_tokens = []
    parse = nlp(sentence)
    for idx, token in enumerate(parse):
        if token.text in female_names in female_names: 
            capitalized_tokens.append(token.text.capitalize())
        elif token.text in male_names or token.text[:-1] in male_names: 
            capitalized_tokens.append(token.text.capitalize())
        elif token.pos_ == 'NOUN' or idx == 0:
            capitalized_tokens.append(token.text.capitalize())
        else:
            capitalized_tokens.append(token.text)
    return ' '.join(capitalized_tokens)




def perturb_possessive_genitive(tokens: List[str], tags: List[str]) -> Tuple[List[str], List[str], bool]:
    sentence = ' '.join(tokens)
    capitalized_sentence = capitalize_sentence(sentence)
    perturbed_tokens = tokens.copy()
    perturbed_tags = tags.copy()
    parse = nlp(sentence)
    stanza_parse = stanza_nlp(sentence)
    replaced = False 
    
    ne_gen_idx = -1
    
    
    for i, token in enumerate(parse):
        if token.tag_ == 'NE':
            info = token.morph.get('Case')
            if info:
                if info[0] == 'Gen':
                    ne_gen_idx = i

    if ne_gen_idx >=0:
        
        try:
            token = parse[ne_gen_idx].text
            dative_word = derbi(token,{'Case': 'Dat'}, [0])  
            if token[:-1].lower() in female_names or token[:-1].lower() in male_names: 
                dative_word = token[:-1]
            morph = parse[ne_gen_idx+1].morph.to_dict()



            if morph['Case'] == 'Nom':
                feats = stanza_parse.sentences[0].words[ne_gen_idx+1].feats
                stanza_morph = {}

                parts = feats.split('|')
                result_dict = {}

                for part in parts:
                    key, value = part.split('=')
                    stanza_morph[key] = value


                for key, value in stanza_morph.items():
                    morph[key] = value

            if parse[ne_gen_idx+1].dep_ == 'pd':
                morph['Case'] = 'Nom'


            if token.lower() in female_names or token[:-1].lower() in female_names: 
                poss_word = derbi('ihre', morph, [0]) 
            elif token.lower() in male_names or token[:-1].lower() in male_names: 
                poss_word = derbi('seine', morph, [0])                 
            else:
                return tokens, tags, replaced  


            if tokens[ne_gen_idx].istitle():
                perturbed_tokens[ne_gen_idx] = dative_word.capitalize()
            else:
                perturbed_tokens[ne_gen_idx] = dative_word
            perturbed_tokens.insert(ne_gen_idx+1, poss_word)
            perturbed_tags.insert(ne_gen_idx+1, 'O')
            replaced = True

            return perturbed_tokens, perturbed_tags, replaced
        except:
            return tokens, tags, replaced
        
    else:
        return tokens, tags, replaced
     
     
    
###################### Article before personal names  ######################

def perturb_article_before_personal_name(tokens: List[str], tags: List[str]) -> Tuple[List[str], List[str], bool]:
    sentence = ' '.join(tokens)
    perturbed_tokens = tokens.copy()
    perturbed_tags = tags.copy()
    replaced = False
    
    parse = nlp(sentence)
    stanza_parse = stanza_nlp(sentence)
    
    morph = [[token.morph.get('Gender'), token.morph.get('Case'), token.tag_, token.head] for token in  parse ]

    perturbed_tokens = tokens.copy()
    perturbed_tags = tags.copy()
    

    for i, (token, tag) in enumerate(zip(tokens, tags)):
        if tag in person_tags:
            
        
            person_gender, person_case, person_tag, person_head = morph[i]
            
            try:
                feats = stanza_parse.sentences[0].words[i].feats
                stanza_morph = {}

                parts = feats.split('|')
                result_dict = {}

                for part in parts:
                    key, value = part.split('=')
                    stanza_morph[key] = value
            except:
                stanza_person_case  = person_case
            
            try:
                stanza_person_case = stanza_morph['Case']
            except KeyError:
                stanza_person_case  = person_case

            
            if token.lower() in female_names or token[:-1].lower() in female_names:
                person_gender = ["FEM"]
                
            if token.lower() in male_names or token[:-1].lower() in male_names:
                person_gender = ["MASC"]
                
            if person_gender == 'Neut' or  person_gender == []:
                person_gender = ['MASC']
            
            if person_tag == 'PPER':
                continue
            
            if person_case != stanza_person_case:
                person_case = [stanza_person_case]
            
            if person_head.text in prepositions_dict["dat"] or tokens[i-1] in prepositions_dict["dat"]:
                person_case = ['Dat']
                
            if person_head.text in prepositions_dict["acc"] or tokens[i-1] in prepositions_dict["acc"]:
                person_case = ['Acc']

            if person_case == []:
                person_case = ['Nom']
            

            
            person_case = person_case[0].upper()
            person_gender = person_gender[0].upper()
            
            
            article = articles_dict[f'{person_gender}.{person_case}']['de'][0]
            
            if i == 0:
                article = article.capitalize()
            
            if article == "dem" and tokens[i-1] == "von":
                perturbed_tokens[i-1] = "vom"
                replace = True
            
            else:
            
                perturbed_tokens.insert(i, article)    
                perturbed_tags.insert(i, 'O')
                replaced = True

    return perturbed_tokens,  perturbed_tags, replaced


###################### wie or als wie used to mark comparative constructions ######################

def perturb_als_in_comparative_constructions(tokens: List[str], tags: List[str]) -> Tuple[List[str], List[str], bool]:
    
    matcher = Matcher(nlp.vocab)
    pattern = [{"POS": "ADV"}, {"LOWER":"als"}]
    matcher.add("AdjectiveAls", [pattern])

    perturbed_tokens = tokens.copy()
    perturbed_tags = tags.copy()

    sentence = ' '.join(tokens)
    perturbed_tokens = tokens.copy()
    perturbed_tags = tags.copy()
    replaced = False
    
    parse = nlp(sentence)
    matches = matcher(parse)

    for match_id, start, end in matches:
        string_id = nlp.vocab.strings[match_id]  # Get string representation
        span = parse[start:end]  # The matched span

        perturbed_tokens[end-1] = 'wie'
        
        # perturbed_tokens.insert(end, 'wie')
        # perturbed_tags.insert(end, 'O')
        replaced = True

        
    return perturbed_tokens, perturbed_tags, replaced


###################### Emphatic double article ######################

def perturb_double_article(tokens: List[str], tags: List[str]) -> Tuple[List[str], List[str], bool]:
    
    sentence = ' '.join(tokens)
    perturbed_tokens = tokens.copy()
    perturbed_tags = tags.copy()
    replaced = False

    parse = nlp(sentence)
    
    matcher = Matcher(nlp.vocab)
    pattern1 = [{"LEMMA": "ein"}, {"LOWER":"so"}, {"POS":"ADJ"}]
    pattern2 = [{"LEMMA": "ein"}, {"LOWER":"sehr"}, {"POS":"ADJ"}]
    pattern3 = [{"LEMMA": "ein"}, {"LOWER":"ganz"}, {"POS":"ADJ"}]
    pattern4 = [{"LEMMA": "ein"}, {"LOWER":"recht"}, {"POS":"ADJ"}]
    pattern5 = [{"LEMMA": "ein"}, {"LOWER":"viel"}, {"POS":"ADJ"}]
    pattern6 = [{"LEMMA": "ein"}, {"LOWER":"groß"}, {"POS":"ADJ"}]
    pattern7 = [{"LEMMA": "ein"}, {"LOWER":"wenig"}, {"POS":"ADJ"}]
    
    # pattern8 = [{"LEMMA": "ein"}, {"POS":"ADV"}, {"POS":"ADJ"}]

    matcher.add("DoubleArticle", [pattern1, pattern2, pattern3, pattern4,
                                 pattern6, pattern5, pattern7])
                                 # pattern6])

    matches = matcher(parse)


    for i, (match_id, start, end) in enumerate(matches):
        string_id = nlp.vocab.strings[match_id]  # Get string representation
        span = parse[start:end]  # The matched span
        
        perturbed_tokens.insert(start+1+i, tokens[start-1]) 
        perturbed_tags.insert(start+1+i, tags[start-1])
        
        replaced = True 
        

    return perturbed_tokens, perturbed_tags, replaced

###################### Discourse & Word Order ######################

###################### Swapped family and given names ######################

def perturb_swap_name(tokens: List[str], tags: List[str]) -> Tuple[List[str], List[str], bool]:       
    
    perturbed_tokens = tokens.copy()
    perturbed_tags = tags.copy()
    replaced = False
    cur_tag = ""

    per_spans = []
    current_span = None
    
    sentence = " ".join(tokens)

    for i, tag in enumerate(tags):
        if tag in person_tags:
            cur_tag = tag
            if current_span:
                per_spans.append(current_span)
            current_span = {'start': i, 'end': i + 1}
        elif tag == cur_tag.replace('B', 'I') and current_span:
            current_span['end'] = i + 1
        else:
            if current_span:
                if is_ne(sentence, current_span):
                    per_spans.append(current_span)
            current_span = None

    if current_span:
        per_spans.append(current_span)

    extracted_spans = []
    
    for span in per_spans:
        start, end = span['start'], span['end']
        try:
            perturbed_tokens[start:end] = [tokens[end-1]] + tokens[start:end-1]
            perturbed_tags = [tags[end-1]] + tags[start:end-1]
        except:
            pass
    
    replaced = tokens != perturbed_tokens
        
    return perturbed_tokens, perturbed_tags, replaced

###################### Obligatory denn in questions  ######################



def check_is_question(tokens: List[str]) -> bool:
    if tokens[-1] == '?':
        return True
    if tokens[0] in question_words or tokens[0] in auxiliary_verbs:
        return True
    return False

def perturb_denn_in_questions(tokens: List[str], tags: List[str]) -> Tuple[List[str], List[str], bool]:

    replaced = False
    if not check_is_question(tokens):
        return tokens, tags, replaced
    
    perturbed_tokens = tokens.copy()
    perturbed_tags = tags.copy()
    
    
    if tokens[0].lower() == "habe" and tokens[1] == "ich":
        perturbed_tokens.insert(2, "denn")
        perturbed_tags.insert(2, "O")
        return perturbed_tokens, perturbed_tags, replaced
        
    sentence = ' '.join(tokens)
    parse = nlp(sentence)
    dep = [token.dep_ for token in parse]
    root_id = dep.index('ROOT')

    perturbed_tokens.insert(root_id+1, 'denn')
    perturbed_tags.insert(root_id+1, 'O')

    replaced = True

    return perturbed_tokens, perturbed_tags, replaced


    
    
###################### Raised auxiliary/modal in 2-verb clusters ######################

def perturb_2_verb_clusters(tokens: List[str], tags: List[str]) -> Tuple[List[str], List[str], bool]:
  
    
    perturbed_tokens = tokens.copy()
    perturbed_tags = tags.copy()
    replaced = False
    
    sentence = ' '.join(tokens)
    
    if "zu tun haben" in sentence:
        return perturbed_tokens, perturbed_tags, replaced
    

    
    
    parse = nlp(sentence)

    matcher = Matcher(nlp.vocab)
    pattern = [{"POS": "VERB"}, {"POS":"AUX"}]
    matcher.add("2VerbCluster", [pattern])

    matches = matcher(parse)

    for match_id, start, end in matches:
        string_id = nlp.vocab.strings[match_id]  # Get string representation
        span = parse[start:end]  # The matched span
        
        try:
            perturbed_tokens[start] = tokens[start+1]
            perturbed_tokens[start+1] = tokens[start]

            perturbed_tags[start] = tags[start+1]
            perturbed_tags[start+1] = tags[start]

            replaced = True 
        except:
            pass

    return perturbed_tokens, perturbed_tags, replaced

###################### Tense & Aspect ######################

###################### Progressive construction with am  ######################

def check_umzu(sentence: str) -> bool:
    if "um" in sentence and "zu" in sentence:
        um_index = sentence.find("um")
        zu_index = sentence.find("zu")
        if um_index < zu_index:
            return True
        else:
            return False
    else:
        return False
    

def perturb_am_infinitive(tokens: List[str], tags: List[str]) -> Tuple[List[str], List[str], bool]:
    sentence = ' '.join(tokens)
    perturbed_tokens = tokens.copy()
    perturbed_tags = tags.copy()
    
    replaced = False
    
    if tokens[-1] == '?' or tokens[0].endswith('e'):
        return perturbed_tokens,  perturbed_tags, replaced
   
    if check_umzu(sentence):
        return perturbed_tokens, perturbed_tags, replaced
    
    for question_word in question_words:
        if question_word in tokens:
            return perturbed_tokens, perturbed_tags, replaced 
        

    parse = nlp(sentence)
    deps = [token.dep_ for token in parse]
    
    for token in parse:
        if token.dep_ == "ROOT" and token.pos_ == "NOUN":
            return perturbed_tokens, perturbed_tags, replaced 

    if 'rc' in deps:
        return perturbed_tokens,  perturbed_tags, replaced
    
    for i, token in enumerate(parse):
        
        if token.tag_ == 'VVFIN':
            if i == 0:
                continue
            try:
                lemma = token.lemma_
                
                if lemma in modal_verbs:
                    replaced = False
                    return perturbed_tokens,  perturbed_tags, replaced
                
                perturbed_tokens[i] = derbi('sein', token.morph.to_dict(), [0]) 
                perturbed_tags[i] = 'O'
                
                if perturbed_tokens[-1] != '.':
                    perturbed_tags += ['O', tags[i]]
                    perturbed_tokens += ['am',lemma]
                else:
                    perturbed_tokens.pop(-1)
                    perturbed_tags += ['O', tags[i], 'O']
                    perturbed_tokens += ['am',lemma, '.']
                    
                replaced = True
            except:
                replaced = False
                pass
    

    
    return perturbed_tokens,  perturbed_tags, replaced

###################### Adverbs & Prepositons ######################

###################### Splitting of pronominal adverbs  ######################


def perturb_da(tokens: List[str], tags: List[str]) -> Tuple[List[str], List[str], bool]:
    
    perturbed_tokens = tokens.copy()
    perturbed_tags = tags.copy()
    
    sentence = ' '.join(tokens)
    
    replaced = False

    for word in dawords:
        if word in tokens or word.capitalize() in tokens or word.upper() in tokens:

            parse = nlp(sentence)
            

            gram_info = [(token.i, token.text, token.tag_, token.dep_) for token in parse if token.text.lower() == word]
            if gram_info[0][2] == 'cp':
                return perturbed_tokens,  perturbed_tags, replaced

            da, preposition = 'da', word[2:]
            da_idx = gram_info[0][0]
            
            random_number = random.randint(0, 2)
            
            if random_number == 0:
                perturbed_tokens[da_idx] = da
                if perturbed_tokens[-1] == '.':
                    perturbed_tokens.insert(-1, preposition)
                else:
                    perturbed_tokens.append(preposition)
                perturbed_tags.append('O')
                replaced = True 
            elif random_number == 1:
                perturbed_tokens[da_idx] = da
                if perturbed_tokens[-1] == '.':
                    perturbed_tokens.insert(-1, preposition)
                else:
                    perturbed_tokens.append(preposition)
                perturbed_tags.append('O')
                replaced = True 
            elif random_number == 2:
                perturbed_tokens[da_idx] =  da+word 
                replaced = True 
                
    return perturbed_tokens, perturbed_tags, replaced



###################### Directive preposition auf  ######################

def perturb_nach(tokens: List[str], tags: List[str]) -> Tuple[List[str], List[str], bool]:
    
    perturbed_tokens = tokens.copy()
    perturbed_tags = tags.copy()
    
    sentence = ' '.join(tokens)

    replaced = False

    parse = nlp(sentence)

    matcher = Matcher(nlp.vocab)
    pattern = [{"LOWER": "nach"}, {"TAG":"NE"}]
    matcher.add("LocationPreposition", [pattern])

    matches = matcher(parse)

    for match_id, start, end in matches:
        try:
            string_id = nlp.vocab.strings[match_id]  # Get string representation
            span = parse[start:end]  # The matched span
            perturbed_tokens[start] = 'auf'
            replaced = True 
        except:
            pass

    return perturbed_tokens,  perturbed_tags, replaced


###################### Locative preposition zu  ######################


def perturb_in(tokens: List[str], tags: List[str]) -> Tuple[List[str], List[str], bool]:
    
    perturbed_tokens = tokens.copy()
    perturbed_tags = tags.copy()
    
    sentence = ' '.join(tokens)

    replaced = False

    parse = nlp(sentence)

    matcher = Matcher(nlp.vocab)
    pattern = [{"LOWER": "in"}, {"TAG":"NE"}]
    matcher.add("LocationPreposition", [pattern])

    matches = matcher(parse)

    for match_id, start, end in matches:
        string_id = nlp.vocab.strings[match_id]  # Get string representation
        span = parse[start:end]  # The matched span
        perturbed_tokens[start] = "zu"
        perturbed = replaced 

    return perturbed_tokens,  perturbed_tags, replaced


###################### Negation ######################

###################### Negative concord  ######################

def perturb_negative_concord(tokens: List[str], tags: List[str]) -> Tuple[List[str], List[str], bool]:
    perturbed_tokens = tokens.copy()
    perturbed_tags = tags.copy()
    sentence = ' '.join(tokens)    
    replaced = False

    parse = nlp(sentence)

    gram_info = [(chunk.text, chunk.end) for chunk in parse.noun_chunks if chunk.text.startswith('kein')]
    if gram_info:
        head_idx = gram_info[0][1]

        perturbed_tokens.insert(head_idx,'nicht')
        perturbed_tags.insert(head_idx,'O')
        replaced = True
        
    return perturbed_tokens, perturbed_tags, replaced

###################### Relativization ######################

###################### Relative pronoun  ######################


def perturb_relative_pronoun(tokens: List[str], tags: List[str]) -> Tuple[List[str], List[str], bool]:
    perturbed_tokens = tokens.copy()
    perturbed_tags = tags.copy()
    sentence = ' '.join(tokens)
    replaced = False

    parse = nlp(sentence)

    matcher = Matcher(nlp.vocab)
    pattern = [{"TAG": "NN"},  {"TAG": "$,"}, {"TAG":"PRELS"}]
    matcher.add("WoRelativePronoun", [pattern])

    matches = matcher(parse)

    for match_id, start, end in matches:
        string_id = nlp.vocab.strings[match_id]  # Get string representation
        span = parse[start:end]  # The matched span
        perturbed_tokens[end-1] = 'wo'
        replaced = True

    return perturbed_tokens,  perturbed_tags, replaced


###################### Complementation ######################

###################### Existential clause ######################


def perturb_existential_clause(tokens: List[str], tags: List[str]) -> Tuple[List[str], List[str], bool]:
    perturbed_tokens = tokens.copy()
    perturbed_tags = tags.copy()
    
    sentence = ' '.join(tokens)
    perturbed_sentence = ' '.join(tokens)
    
    replaced = False

    parse = nlp(sentence)

    matcher = Matcher(nlp.vocab)
    pattern = [{"LOWER": "es"}, {"LOWER":"gibt"}]
    matcher.add("EsGibtConstuction", [pattern])

    matches = matcher(parse)

    for match_id, start, end in matches:
        string_id = nlp.vocab.strings[match_id]  # Get string representation
        span = parse[start:end]  # The matched span
        perturbed_tokens[end-1] = "hat"
        replaced = True 

    return perturbed_tokens, perturbed_tags, replaced




###################### VERB MOPHOLOGY ######################

###################### Schwa elision at the end of 1.sg.pres verbs ######################

def perturb_schwa_elision_in_verbs(tokens: List[str], tags: List[str]) -> Tuple[List[str], List[str], bool]:
    perturbed_tokens = tokens.copy()
    perturbed_tags = tags.copy()
    
    sentence = " ".join(tokens)
    parse = nlp(sentence)
                
    replaced = False
    
    verbs = ["habe"]
    
    for i, token in enumerate(parse):
        if (token.pos_ == "VERB" and token.text.endswith("e")) or (token.text in verbs):
            r = random.randint(0, 10)
            if r % 2 == 0:
                perturbed_tokens[i] = token.text[:-1]
            else:
                perturbed_tokens[i] = token.text[:-1] + "'"
            replaced = True
            
    return perturbed_tokens, perturbed_tags, replaced


###################### Constructions with tun in imperatives ######################


def perturb_tun_imperativ(tokens: List[str], tags: List[str]) -> Tuple[List[str], List[str], bool]:
    
    import string 
    
    sentence = " ".join(tokens)
    parse = nlp(sentence)
    replaced = False
        
    pos_tags = [token.pos_ for token in parse]
    
    is_imperativ = False
    
    if pos_tags[0] == 'VERB' and tokens[0].endswith('e') or tokens[0].endswith('en') or any(char in tokens[0] for char in "äöü"):
        is_imperativ = True

    
    if is_imperativ == False:
        return tokens, tags, replaced
        
    verb = conjugate(tokens[0].lower(), INFINITIVE)
    if tokens[0].lower() == 'schalte':
        verb = 'schalten'

    perturbed_tokens = tokens.copy()
    perturbed_tags = tags.copy()
    
    try:
        for ch in parse[0].children:
            if ch.pos_ == 'ADP':
                if ch.i < len(tokens) - 2:
                    continue
                verb = ch.text + verb
                perturbed_tokens.pop(ch.i)
                perturbed_tags.pop(ch.i)
    except:
        pass
    


    if tokens[0].endswith('e'):
        r = random.randint(0, 10)
        if r % 2 == 0:
            new_verb = 'Tue'
        else:
            new_verb = 'Tu'
            
    elif tokens[0].endswith('n'):
        new_verb = 'Tuen'
        
    else: 
        new_verb = 'Tu'
        
    if tokens[-1] in string.punctuation:
        perturbed_tokens.insert(-1, verb)
    else:
        perturbed_tokens.append(verb)
    perturbed_tokens = [new_verb] + perturbed_tokens[1:] 
    replaced = True 
    
    perturbed_tags = perturbed_tags + ['O']

    
    return perturbed_tokens, perturbed_tags, replaced

###################### PRONOUNS ######################


###################### Contracted verb and pronoun ######################

def perturb_contracted_verb_and_pronoun(tokens: List[str], tags: List[str]) -> Tuple[List[str], List[str], bool]:
    perturbed_tokens = tokens.copy()
    perturbed_tags = tags.copy()
    sentence = ' '.join(tokens)
    
    replaced = False

    parse = nlp(sentence)
    
    matcher = Matcher(nlp.vocab)
    pattern1 = [{"POS": "AUX"}, {"LOWER":"es"}]
    pattern2 = [{"POS": "VERB"}, {"LOWER":"es"}]
    matcher.add("EsVerb", [pattern1, pattern2])
    matches = matcher(parse)
    if len(matches) == 0:
        return tokens,  tags, replaced
    for match in matches:
        r = random.randint(0, 10)
        start, end = match[1], match[2]
        if r % 2 == 0:
            perturbed_tokens[match[1]] = perturbed_tokens[match[1]]+"s"
        else:
            perturbed_tokens[match[1]] = perturbed_tokens[match[1]]+"'s"
        perturbed_tokens.pop(match[1]+1)
        perturbed_tags.pop(match[1]+1)
        replaced = True
        
    return perturbed_tokens, perturbed_tags, replaced
