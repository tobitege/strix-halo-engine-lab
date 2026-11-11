"""Generate a synthetic tokenizer for host HTTP tests; no model assets or fidelity claims."""
import sys
import json
from pathlib import Path
v = {f'<unused-{i}>': i for i in range(262144)}
by_id = list(v)
def token(text, i):
    old = by_id[i]
    del v[old]
    v[text] = i
    by_id[i] = text
special = {1:'<eos>',2:'<bos>',106:'<turn|>',107:'<|channel>',108:'<|tool_call>',109:'<tool_call|>',110:'<|tool_response>',111:'<|"|>',112:'<|turn>'}
for i,s in special.items(): token(s,i)
for b in range(256): token(f'<0x{b:02X}>',1000+b)
for b in range(128): token(chr(b),2000+b)
token('\u2581',2200)
merges = []
word = 'thought\nChecking'
for j in range(2,len(word)+1):
    token(word[:j],3000+j)
    merges.append([word[:j-1],word[j-1]])
asset = {'version':'1.0','normalizer':{'type':'Replace','pattern':{'String':' '},'content':'\u2581'},
 'pre_tokenizer':{'type':'Split','pattern':{'String':' '},'behavior':'MergedWithPrevious','invert':False},
 'decoder':{'type':'Sequence','decoders':[{'type':'Replace','pattern':{'String':'\u2581'},'content':' '},{'type':'ByteFallback'},{'type':'Fuse'}]},
 'post_processor':{'type':'TemplateProcessing','single':[{'Sequence':{'id':'A','type_id':0}}],'pair':[{'Sequence':{'id':'A','type_id':0}},{'Sequence':{'id':'B','type_id':1}}],'special_tokens':{}},
 'truncation':None,'padding':None,
 'model':{'type':'BPE','dropout':None,'byte_fallback':True,'ignore_merges':False,'fuse_unk':True,'unk_token':'<unk>','continuing_subword_prefix':None,'end_of_word_suffix':None,'vocab':v,'merges':merges},
 'added_tokens':[{'id':i,'content':s,'single_word':False,'lstrip':False,'rstrip':False,'normalized':False,'special':True} for i,s in special.items()]}
Path(sys.argv[1]).write_text(json.dumps(asset,ensure_ascii=False,separators=(',',':')),encoding='utf-8')
print('Synthetic host-only tokenizer fixture generated')
