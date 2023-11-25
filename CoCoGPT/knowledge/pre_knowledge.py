from pytorch_pretrained_bert import BertTokenizer
import torch

if __name__ == "__main__":
    total_sent = 0
#    max_word = 100
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    f = open("knowledge.txt", "r")
    knowledge = []
    count_dict = {}
    total_word = 0
    while True:
        line = f.readline()
        if not line:
           break

        encoder_input = tokenizer.tokenize(line)
        encoder_input = tokenizer.convert_tokens_to_ids(encoder_input)
        knowledge.extend(encoder_input)
#        print (encoder_input)
    knowledge = torch.LongTensor(knowledge)
    print (knowledge.size())
    torch.save(knowledge, 'knowledge.pth')
        
#        if len(line) > max_word:
#            max_word = len(line)

#        print (line)
#    print (total_sent, max_word)
