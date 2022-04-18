import logging

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)



#Data preparation
from datasets import load_dataset
train_data = load_dataset('narrativeqa', split='train[:1]')
# test_data = load_dataset('narrativeqa', split='test[:10%]')
# eval_data = load_dataset('narrativeqa', split='eval[:10%]')
print(train_data)




# from simpletransformers.question_answering import QuestionAnsweringModel, QuestionAnsweringArgs
#
#
# model_args = QuestionAnsweringModel(
#     n_best_size=2,
#     doc_stride=500,
#     max_answer_length=500,
#     max_query_length=500
# )
#
# model = QuestionAnsweringModel(
#     "xlnet",
#     "xlnet-base-cased",
#     args=model_args,
#     use_cuda=True
# )
#
#
# model.train_model(
#     train_data,
#     output_dir="./result/",
#     show_running_loss=True,
# )
#
# result, model_outputs, wrong_preds = model.eval_model(
#     eval_data,
#     verbose=True,
#     verbose_logging=True,
#     output_dir="./result/",
# )