import sys
from PyQt6.QtWidgets import QApplication, QWidget
from PyQt6 import QtCore, QtGui, QtWidgets
from features import fs_encode
import lightgbm as lgb
import numpy as np

import torch
model_esm, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")

def esm_infer(seq):
    batch_converter = alphabet.get_batch_converter()
    model_esm.eval()  # disables dropout for deterministic results
    # Prepare data (first 2 sequences from ESMStructuralSplitDataset superfamily / 4)
    data = [
        ("tmp", seq),
    ]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    batch_tokens = batch_tokens
    # batch_tokens = batch_tokens

    with torch.no_grad():
        results = model_esm(batch_tokens, repr_layers=[33], return_contacts=True)
    token_representations = results["representations"][33]
    token_representations = token_representations.detach().cpu().numpy()
    # print(token_representations.shape)
    token_representations = token_representations[0][1:-1,:]
    # (7, 1280)
    return token_representations.sum(axis=0)

'''
pip install transformers
pip install SentencePiece
'''

from transformers import T5Tokenizer, T5Model,T5EncoderModel
import re

tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_uniref50',do_lower_case=False)
model_t5 = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")

def T5_infer(seq):
    sequences_Example = [" ".join(list(seq))]
    sequences_Example = [re.sub(r"[UZOB]", "X", sequence) for sequence in sequences_Example]
    ids = tokenizer.batch_encode_plus(sequences_Example,
                                      add_special_tokens=True,
                                      padding=False)

    input_ids = torch.tensor(ids['input_ids'])
    attention_mask = torch.tensor(ids['attention_mask'])

    input_ids = input_ids
    attention_mask = attention_mask

    with torch.no_grad():
        # embedding = model(input_ids=input_ids,attention_mask=attention_mask,decoder_input_ids=None)
        embedding = model_t5(input_ids=input_ids,
                          attention_mask=attention_mask,
                          # decoder_input_ids=input_ids,
                          )

    # For feature extraction we recommend to use the encoder embedding
    encoder_embedding = embedding.last_hidden_state[0,:-1].detach().cpu().numpy()
    return encoder_embedding.sum(axis=0)


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(784, 666)
        font = QtGui.QFont()
        font.setPointSize(16)
        Form.setFont(font)
        self.plainTextEdit = QtWidgets.QPlainTextEdit(parent=Form)
        self.plainTextEdit.setGeometry(QtCore.QRect(150, 100, 511, 261))
        self.plainTextEdit.setObjectName("plainTextEdit")
        self.pushButton = QtWidgets.QPushButton(parent=Form)
        self.pushButton.setGeometry(QtCore.QRect(150, 370, 91, 41))
        self.pushButton.setIconSize(QtCore.QSize(32, 32))
        self.pushButton.setObjectName("pushButton")

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.plainTextEdit.setPlainText(_translate("Form", "Input Fasta format peptide sequence"))
        self.pushButton.setText(_translate("Form", "submit"))


class Login(QWidget):
    def __init__(self):
        super().__init__()

        # use the Ui_login_form
        self.ui = Ui_Form()
        self.ui.setupUi(self)

        self.ui.pushButton.clicked.connect(self.write_res)

        # show the login window
        self.show()
        self.model_fs = lgb.Booster(model_file="./models/model.fs")
        self.model_tr = lgb.Booster(model_file="./models/model.transformer")


    def get_data(self):
        id_seq_dict ={}
        seq = self.ui.plainTextEdit.toPlainText()
        seq_list = seq.strip().split("\n")
        id = ""
        for i in range(len(seq_list)):
            if (i%2)==0:
                id = seq_list[i]
            else:
                id_seq_dict[id]=seq_list[i]
        return id_seq_dict

    def encode(self):
        id_encode_dict= {}
        for id,seq in self.get_data().items():
            id = id
            seq = seq
            fsvec = fs_encode(seq)
            res_esm = esm_infer(seq)
            res_t5 = T5_infer(seq)
            res2 = np.concatenate([res_t5,res_esm])
            id_encode_dict[id] = [fsvec,res2]
        return id_encode_dict

    def predict(self):
        id_encode_dict = self.encode()
        ress = {}
        for id,vec in id_encode_dict.items():
            vec_fs = np.array(vec[0])[np.newaxis,:]
            vec_ts = np.array(vec[1])[np.newaxis,:]
            p1 = self.model_fs.predict(vec_fs).flatten()[0]
            p2 = self.model_tr.predict(vec_ts).flatten()[0]
            t = 0.4
            if p1>0.4 and p2>0.4:
                ress[id] = max([p1,p2])
            elif p1<0.4 and p2<0.4:
                ress[id]=min([p1,p2])
            else:
                ress[id]=np.mean([p1,p2])
        return ress

    def write_res(self):
        f = open("predict_results.csv","w")
        ress_dict = self.predict()
        for id,ps in ress_dict.items():
            tmp = id+","+str(ps)+"\n"
            f.write(tmp)
        f.close()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    login_window = Login()
    sys.exit(app.exec())