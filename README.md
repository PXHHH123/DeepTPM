# **DeepTPM**

**DeepTPM**: Deep learning ensemble model for TCR-pMHC binding specificity prediction

-------------------

This repository contains the code and the data to train and test **DeepTPM** model.

 Contact: [1299890870@qq.com](mailto:1299890870@qq.com); [solfix123@163.com](mailto:solfix123@163.com)

## Project Structure

```
Project Directory Structure:
├── TCR_encoding                # Encode TCR data using a fine-tuned ProteinBERT
│   ├── TCR_encoder             # Scripts for fine-tuning ProteinBERT and extracting features from TCR data
│   ├── TCR_result              # TCR feature extraction results ProteinBERT
│   │   ├── TCR_encoding        # Feature extraction results of TCR sequences using the DeepTPM model
│   │   ├── TCR_encoding_retrain # Feature extraction results after retraining the ProteinBERT model
│   ├── proteinbert             # Stores related parameters of the ProteinBERT model
│   ├── data.csv                # Original data for fine-tuning ProteinBERT
│
├── PMHC_encoding               # pMHC feature extraction model for peptide-MHC
│   ├── PMHC-encoder.jpynb      # Script for extracting features from pMHC data
│   ├── pMHC_feature_extraction_model_train.jpynb  # Script for training the pMHC feature extraction model
│   ├── pMHC_feature_extraction_model.pth          # Trained pMHC feature extraction model
│   ├── data                    # Test data for DeepTPM (proteins.txt)
│   ├── proteins.txt            # Data for training the pMHC feature extraction model
│   ├── PMHC_result             # pMHC feature extraction results
│
├── DeepTPM_main.jpynb          # Main script to build the pMHC feature extraction model
│
├── Prediction results          # Prediction results for 9 test datasets
│
└── README.md                   # Project overview, including directory structure and usage instructions

```

### Data Availability

- The one million CDR3 sequences used to fine-tune ProteinBERT can be downloaded from https://github.com/hliulab/atmtcr/. 
- The PROTEINS training dataset can be downloaded at https://github.com/uci-cbcl/HLA-bind.
- The large dataset (32,044 samples) and the small dataset (619 samples) can be ---downloaded from https://github.com/tianshilu/pMTnet.
- The 10 datasets used for generalization ability testing of epiTCR can be downloaded from https://github.com/ddiem-ri-4D/epiTCR.

## Usage

#### Input file format

The input files are csv files in the following format:

```
	CDR3			Antigen	        HLA		label
1	CASSPNGDRVFDQPQHF	GILGFVFTL	A*02:01		  0
2	CASSVRQEPYNEQF		NLVPMVATV	A*02:01	          0
3	CASSPRGQGYMNTEAFF	GLCTLVAML	A*02:01	          0
4  	CSARDLDRDGTDTQYF	AVFDRKSDAK	A*11:01		  1
5	CASSLVAGGQETQYF		KLGGALQAK	A*03:01		  1
6	CAWSWSGGGTGELFF		NLVPMVATV	A*02:01	          0
......
```

#### TCR feature extraction

```python
TCR feature extraction-->TCR_encoding-->TCR_encoder.jpynb  
 
---df = load_hla_dataframe('../PMHC_encodeing/data/epiTCR_data/train.csv')
---encoded_x_test = input_encoder.encode_X(list(df.CDR3), 34)
---feature_extraction_model = Model(inputs=finetuned_model.input, outputs=intermediate_output1)
---new_features = feature_extraction_model.predict(encoded_x_test)
```

#### pMHC feature extraction

```
pMHC feature extraction-->PMHC_encoding-->PMHC-encoder.jpynb 

---df = load_hla_dataframe("data/epiTCR_data/test01.csv")
---df1 = get_hla_subtype(df)
---device = "cuda"
---Peptide, lengths, hla = code(df1)
---all_y_preds = Feature_extraction(Peptide, lengths, hla)
```

#### lightGBM：Feature fusion and classification

```python
jupyter notebook-->DeepTPM_main.jpynb

---data_1 = torch.load("PMHC_encodeing/PMHC_result/pmhc_large_data_encoding.pt")
---data_2 = torch.load("TCR_encodeing/TCR_result/TCR_encoding/TCR_large_encoding.pt")
---df = load_hla_dataframe('PMHC_encodeing/data/large_data.csv')

---train_and_test(df, data_1, data_2, params)
```

The output will be like

```
{'AUC': 0.944, 'Accuracy': 0.899, 'Sensitivity': 0.955, 'Specificity': 0.844, 'Threshold': 0.5, 'F1': 0.902}
```

#### DeepTPM  generalization ability test

```python
jupyter notebook-->DeepTPM_main.jpynb

############### Load Training Dataset and Train Model ######################

---numpy_array = np.load('PMHC_encodeing/PMHC_result/epiTCR_data_encoding/pmhc_train_encoding.npy')
---data_1 = torch.from_numpy(numpy_array)
---data_2 = torch.load("TCR_encodeing/TCR_result/TCR_encoding/TCR_train.pt")
---df = load_hla_dataframe('PMHC_encodeing/data/epiTCR_data/train.csv')

---bst = train(df, data_1, data_2, params)

############# Load Testing Dataset and Test Model ################

---data_test_1 = torch.load("PMHC_encodeing/PMHC_result/epiTCR_data_encoding/pmhc_test01_encoding.pt")
---data_test_2 = torch.load("TCR_encodeing/TCR_result/TCR_encoding/TCR_test01.pt")
---df_test = load_hla_dataframe('PMHC_encodeing/data/epiTCR_data/test01.csv')

---test(df_test, data_test_1, data_test_2, bst)


```

The output will be like

```
{'AUC': 0.969, 'Accuracy': 0.888, 'Sensitivity': 0.935, 'Specificity': 0.884, 'Threshold': 0.5, 'F1': 0.602}
```



## Python and essential packages

```
python         3.10.9
numpy          1.23.5
pandas         1.5.3
torch       2.0.1+cu117
tensorflow     2.16.0
keras          3.3.3
```

##### Device Specifications for Fine-Tuning ProteinBERT

```
Fine-tuning of ProteinBERT was conducted using an NVIDIA GeForce RTX 4090 GPU.
```

