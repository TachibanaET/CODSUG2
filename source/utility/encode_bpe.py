import numpy as np
import jaconv
import re

class BPEEncoder_ja:
  def __init__(self, bpe, emoji):
    self.bpe = bpe
    self.emoji = emoji
    self.maxlen = np.max([len(w) for w in self.bpe])
    self.content_repatter1 = re.compile(r"(https?|ftp)(:\/\/[-_\.!~*\'()a-zA-Z0-9;\/?:\@&=\+$,%#]+)")
    self.content_repatter2 = re.compile(r"[A-Za-z0-9\._+]*@[\-_0-9A-Za-z]+(\.[A-Za-z]+)*")
    self.content_repatter3 = re.compile(r'[\(]{0,1}[0-9]{2,4}[\)\-\(]{0,1}[0-9]{2,4}[\)\-]{0,1}[0-9]{3,4}')
    self.content_repatter4 = re.compile(r"([12]\d{3}[/\-年])*(0?[1-9]|1[0-2])[/\-月]((0?[1-9]|[12][0-9]|3[01])日?)*(\d{1,2}|:|\d{1,2}時|\d{1,2}分|\(日\)|\(月\)|\(火\)|\(水\)|\(木\)|\(金\)|\(土\))*")
    self.content_repatter5 = re.compile(r"(明治|大正|昭和|平成|令和)\d{1,2}年(0?[1-9]|1[0-2])月(0?[1-9]|[12][0-9]|3[01])日(\d{1,2}|:|\d{1,2}時|\d{1,2}分|\(日\)|\(月\)|\(火\)|\(水\)|\(木\)|\(金\)|\(土\))*")
    self.content_repatter6 = re.compile(r'((0|[1-9]\d*|[1-9]\d{0,2}(,\d{3})+)*億)*((0|[1-9]\d*|[1-9]\d{0,2}(,\d{3})+)*万)*((0|[1-9]\d*|[1-9]\d{0,2}(,\d{3})+)*千)*(0|[1-9]\d*|[1-9]\d{0,2}(,\d{3})+)*(千円|万円|千万円|円|千ドル|万ドル|千万ドル|ドル|千ユーロ|万ユーロ|千万ユーロ|ユーロ)+(\(税込\)|\(税抜\)|\+tax)*')

  def __len__(self):
    return len(self.bpe)

  def clean_text(self, content):
    content = jaconv.z2h(content, kana=False, digit=True, ascii=True)
    content = self.content_repatter1.sub("<URL>" ,content)
    content = self.content_repatter2.sub("<EMAIL>" ,content)
    content = self.content_repatter3.sub("<TEL>" ,content)
    content = self.content_repatter4.sub("<DATE>" ,content)
    content = self.content_repatter5.sub("<DATE>" ,content)
    content = self.content_repatter6.sub("<PRICE>" ,content)
    return content

  def encode(self, text, clean=False, padding=False, max_len=128):
    text = text.replace(' ', '<SP>')
    text = text.replace('　', '<SP>')
    text = text.replace('\r\n', '<BR>')
    text = text.replace('\n', '<BR>')
    text = text.replace('\r', '<BR>')
    text = text.replace('\t', '<TAB>')
    text = text.replace('—', 'ー')
    text = text.replace('−', 'ー')
    for k,v in self.emoji['emoji'].items():
      if k in text:
        text = text.replace(k, v)
    if clean:
      text = self.clean_text(text)
    pos = 0
    result = []
    while pos < len(text):
      bp = False
      end = min(len(text), pos+self.maxlen+1) if text[pos]=='<' else pos+2
      for e in range(end, pos, -1):
        wd = text[pos:e]
        if wd in self.bpe:
          result.append(self.bpe.index(wd))
          pos = e
          bp = True
          break
      if not bp:
        end = pos+1
        wd = text[pos:end]
        for i in wd.encode('utf-8'):
            result.append(self.bpe.index('<|byte%d|>'%i))
        pos = end
    
    attention_mask = [1] * len(result)

    if(padding):
      if len(result) > max_len:
        result = result[:max_len]
        attention_mask = attention_mask[:max_len]
      else:
        attention_mask += [0] * (max_len - len(result))
        result += [0] * (max_len - len(result))
    return {
        'input_ids' : result,
        'attention_mask' : attention_mask
    }
    
  def decode(self, tokens, breakline='\n'):
    words = []
    byte_tokens = []
    for i in tokens:
      word = self.bpe[i]
      if word[:6] == '<|byte' and word[-2:] == '|>':
        byte_tokens.append(int(word[6:-2]))
      else:
        if len(byte_tokens) > 0:
          words.append(bytearray(byte_tokens).decode('utf-8', errors='replace'))
          byte_tokens = []
        if word[:7] == '<|emoji' and word[-2:] == '|>':
          words.append(self.emoji['emoji_inv'][word])
        elif word == '<SP>':
          words.append(' ')
        elif word == '<BR>':
          words.append(breakline)
        elif word == '<TAB>':
          words.append('\t')
        else:
          words.append(word)
    if len(byte_tokens) > 0:
      words.append(bytearray(byte_tokens).decode('utf-8', errors='replace'))
    text = ''.join(words)
    return text


if __name__=='__main__':
  pass