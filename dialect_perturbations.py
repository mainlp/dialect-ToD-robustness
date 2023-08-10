import os
import pandas as pd
import numpy as np
import json
from typing import Dict, List, Tuple

import spacy
from spacy.matcher import Matcher
from spacy.attrs import * 
from somajo import SoMaJo
import random 

from pattern.de import conjugate
from pattern.de import INFINITIVE, PRESENT, SG, SUBJUNCTIVE, PAST, PARTICIPLE

from DERBI.derbi import DERBI


folder_path = "CharSplit"

try:
    os.rmdir(folder_path)
    print(f"Folder '{folder_path}' has been removed.")
except OSError as e:
    print(f"Error: {e}")


resources_path = 'phenomena/resources'

tokenizer = SoMaJo("de_CMC", split_camel_case=True)
nlp = spacy.load('de_core_news_sm')
derbi = DERBI(nlp)

prepositions_with_genitive = [line.strip() for line in open(f'{resources_path}/prepositions_with_genitive.txt').readlines()]
articles_dict = json.load(open(f'{resources_path}/definite_article.json'))
person_tags = ['B-'+line.strip() for line in open(f'{resources_path}/person_tags.txt').readlines()]
question_words = [line.strip() for line in open(f'{resources_path}/question_words.txt').readlines()]
auxiliary_verbs = [line.strip() for line in open(f'{resources_path}/auxiliary_verbs.txt').readlines()]
dawords = [line.strip() for line in open(f'{resources_path}/dawords.txt').readlines()]


    
###################### NOUN GROUPS ######################

###################### von construction instead of genitive ######################

def get_genitive_groups(sentence :str) -> List[str]:
    '''
        get spans of genitive groups 
    '''
    parse = nlp(sentence)
    genitive_groups = []
    
    for token in parse:
        if "Gen" in token.morph.get('Case'):
            genitive_group = " ".join([child.text for child in token.children])
            if genitive_group:
                genitive_groups.append(" ".join([genitive_group,token.text]))
    return genitive_groups


def genitive_group_to_dativ_group(genitive_group: str) -> str:
    '''
        change a genitive group into a dative group
    '''
    info = [token.morph.to_dict() for token in nlp(genitive_group)]
    inflected_info = []
    idx = [i for i in range(len(info))]
    result = []
    for token in info:
    
        token['Case']='Dat'
        inflected_info.append(token)

    dativ_group = derbi(genitive_group, inflected_info, idx)
    for token1, token2 in zip(genitive_group.split(), dativ_group.split()):
        if token1[0].isupper():
            result.append(token2[0].upper() + token2[1:])
        else:
            result.append(token2)
    
    dativ_tokens = dativ_group.split()
    if dativ_tokens[0] == 'dem':
        dativ_group = ['vom'] + result[1:]
    else:
        dativ_group = ['von'] + result
    
    dativ_group = " ".join(dativ_group)
    return dativ_group


def perturb_genitive_to_dativ(tokens: List[str], tags: List[str]) -> Tuple[List[str], List[str], bool]:
    '''
        perturb sentence by changing genitive to dativ
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
        dativ_group = genitive_group_to_dativ_group(genitive_group)
        perturbed_sentence = sentence.replace(genitive_group, dativ_group)
        perturbed_tokens = perturbed_sentence.split()
        replaced = True 
        
    for i, token in enumerate(perturbed_tokens):
        if token in ['von','vom']:
            perturbed_tags.insert(i,'O')
            
    return perturbed_tokens, perturbed_tags, replaced

###################### Possessive dative instead of genitive ######################


def perturb_possesive_genitive(tokens: List[str], tags: List[str]) -> Tuple[List[str], List[str], bool]:
    sentence = ' '.join(tokens)
    perturbed_tokens = tokens.copy()
    perturbed_tags = tags.copy()
    parse = nlp(sentence)
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
            dativ_word = derbi(parse[ne_gen_idx].text,{'Case': 'Dat'}, [0])  
            poss_word = derbi('seine', parse[ne_gen_idx+1].morph.to_dict(), [0])   

            if tokens[ne_gen_idx].istitle():
                perturbed_tokens[ne_gen_idx] = dativ_word.capitalize()
            else:
                perturbed_tokens[ne_gen_idx] = dativ_word
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
    
    morph = [[token.morph.get('Gender'), token.morph.get('Case'), token.tag_] for token in  parse ]

    perturbed_tokens = tokens.copy()
    perturbed_tags = tags.copy()
    

    for i, (token, tag) in enumerate(zip(tokens, tags)):
        if tag in person_tags:
            person_gender, person_case, person_tag = morph[i]
            
            if person_tag == 'PPER':
                continue
            
            if person_gender == 'Neut' or  person_gender == []:
                person_gender = ['MASC']
            if person_case == []:
                person_case = ['Nom']
            
            person_case = person_case[0].upper()
            person_gender = person_gender[0].upper()
            
            article = articles_dict[f'{person_gender}.{person_case}']['de'][0]
            
            if i == 0:
                article = article.capitalize()
            
            perturbed_tokens.insert(i, article)    
            perturbed_tags.insert(i, 'O')
            replaced = True
    
    perturbed_tokens[0]
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

    per_spans = []
    current_span = None

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
                per_spans.append(current_span)
            current_span = None

    if current_span:
        per_spans.append(current_span)

    extracted_spans = []
    for span in per_spans:
        start, end = span['start'], span['end']
        perturbed_tokens[start:end] = [tokens[end-1]] + tokens[start:end-1]
        replaced = True
        
    return perturbed_tokens, perturbed_tags, replaced

###################### Obligatory denn in questions  ######################



def check_is_question(tokens: List[str]) -> bool:
    if tokens[-1] == '?':
        return True
    if tokens[0] in question_words or token[0] in auxiliary_verbs:
        return True
    return False

def perturb_denn_in_questions(tokens: List[str], tags: List[str]) -> Tuple[List[str], List[str], bool]:

    replaced = False
    if not check_is_question(tokens):
        return tokens, tags, replaced
    
    perturbed_tokens = tokens.copy()
    perturbed_tags = tags.copy()
    
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

    if 'rc' in deps:
        return perturbed_tokens,  perturbed_tags, replaced
    
    for i, token in enumerate(parse):
        
        if token.tag_ == 'VVFIN':
            if i == 0:
                continue
            try:
                lemma = token.lemma_
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
        string_id = nlp.vocab.strings[match_id]  # Get string representation
        span = parse[start:end]  # The matched span
        perturbed_tokens[start] = 'auf'
        replaced = True 

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
    
    tokens = sentence.split()
    perturbed_tokens = tokens.copy()

    tags = ['O'] * len(tokens)
    perturbed_tags = tags.copy()
    
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


###################### OTHER ######################


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

###################### Schwa elision at the end of 1.sg.pres verbs ######################

def perturb_schwa_elision_in_verbs(tokens: List[str], tags: List[str]) -> Tuple[List[str], List[str], bool]:
    perturbed_tokens = tokens.copy()
    perturbed_tags = tags.copy()
    replaced = False
    
    for i, token in enumerate(tokens):
        if token == "habe":
            r = random.randint(0, 10)
            if r % 2 == 0:
                perturbed_tokens[i] = "hab"
            else:
                perturbed_tokens[i] = "hab'"
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
       
    for ch in parse[0].children:
        if ch.pos_ == 'ADP':
            verb = ch.text + verb
            perturbed_tokens.pop(ch.i)
            perturbed_tags.pop(ch.i)
    


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