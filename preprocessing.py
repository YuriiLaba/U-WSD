import os

train_id = "1-0hB7A45jbciW_fBEnQ02VqDTf6SBtVB"
eval_id = "1-0wj4UBP0S8JfWLSmjz81Z2BOEu7tNqe"

train_name = "wsd_lemma_homonyms_dataset_triplet_500k_train_95.csv"
eval_name = "wsd_lemma_homonyms_dataset_triplet_500k_eval_5.csv"

os.system("""wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1-7TiJ92Ly3wqwUHqJFY5nL--IdCL8-M-' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1-7TiJ92Ly3wqwUHqJFY5nL--IdCL8-M-" -O 20180506.uk.mova-institute.udpipe && rm -rf /tmp/cookies.txt""")
os.system(f"""wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id={eval_id}' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id={eval_id}" -O {eval_name} && rm -rf /tmp/cookies.txt""")
os.system(f"""wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id={train_id}' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id={train_id}" -O {train_name} && rm -rf /tmp/cookies.txt""")
