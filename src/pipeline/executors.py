
def simple_pipeline_executor(texts, extractor_fn, nlp_pipeline, word_processors=None, sentence_processors=None):

    results = []
    for text in texts:
        results.append(extractor_fn(text, nlp_pipeline, word_processors, sentence_processors))

    return results
