# Experiment 08 -- OCR and Text Understanding

## 1. Experiment Overview

**Model:** `meta-llama/Llama-3.2-11B-Vision-Instruct`\
**Execution timestamp:** 13 November 2026, 10:39:46\
**Hardware:** Tesla T4\
**Quantization:** 4-bit\
**Execution time:** 16.34 seconds\
**Project repository:**
https://github.com/abhinash7729/LMM-Vision-Language-Project

### Objective

To evaluate the model's ability to read text from an input image,
preserve its wording and punctuation, identify headings and important
details, organize extracted text, summarize the content, and report
unclear or unreadable text without inventing missing information.

## 2. Input Image

  Parameter          Value
  ------------------ -----------------------
  Input image        `data/text_image.png`
  Dimensions         416 × 480 pixels
  Number of images   1

## 3. Model Output -- Extracted Text

The model identified the image as a handwritten letter addressed to
Santa Claus. The following transcription reproduces the text returned by
the model:

> Dear Santa,
>
> I have been very good this year and have been thinking of what I would
> like for Christmas.
>
> 1.  lego
> 2.  bicycle
> 3.  puppy
>
> I am very excited for your visit and will leave out some snacks for
> you and your reindeer.
>
> Love.

**Note:** This is the model's returned transcription. The supplied
result does not include a separate human-verified transcription.

## 4. Important Information Identified

  -----------------------------------------------------------------------
  Category                            Information returned by the model
  ----------------------------------- -----------------------------------
  Document type                       Handwritten letter

  Recipient                           Santa Claus

  Occasion                            Christmas

  First requested gift                lego

  Second requested gift               bicycle

  Third requested gift                puppy

  Additional message                  The writer says snacks will be left
                                      for Santa and the reindeer.

  Closing                             "Love."

  Background                          Described as light blue

  Handwriting                         Described as a child's handwriting
  -----------------------------------------------------------------------

The description of the handwriting as belonging to a child is the
model's interpretation, not a fact established by the extracted text.

## 5. Summary of Image Content

According to the model, the image contains a handwritten Christmas
letter on a light blue background. The writer says they have been good
during the year and lists three gifts they would like: Lego, a bicycle,
and a puppy. The letter also expresses excitement about Santa's visit
and says that snacks will be left for Santa and the reindeer.

## 6. Evaluation

  -----------------------------------------------------------------------
  Criterion               Assessment              Evidence from output
  ----------------------- ----------------------- -----------------------
  Document-type           Successful in output    Identified as a letter.
  identification                                  

  Text extraction         Promising               Returned a readable
                                                  transcription.

  Wording preservation    Requires verification   No independent
                                                  reference transcription
                                                  was supplied.

  Punctuation and         Partially demonstrated  Returned list items are
  capitalization                                  lowercase; exact
                                                  image-level accuracy is
                                                  unverified.

  Important-detail        Successful in output    Recipient, occasion,
  identification                                  gift list, and closing
                                                  were identified.

  Text organization       Successful              Text was organized into
                                                  paragraphs and a
                                                  numbered list.

  Summary generation      Successful              A concise summary
                                                  consistent with the
                                                  returned transcription
                                                  was provided.

  Uncertainty reporting   Not demonstrated        No unclear or
                                                  unreadable words were
                                                  explicitly identified.
  -----------------------------------------------------------------------

## 7. Limitations

1.  The result does not include word-level or character-level confidence
    scores.
2.  There is no human-verified transcription, so exact OCR accuracy
    cannot be calculated.
3.  The model did not explicitly state whether any words were unclear or
    unreadable.
4.  The description of the writer as a young child is an interpretation
    based on handwriting.
5.  The returned output is readable, but readability alone does not
    establish character-level accuracy.

## 8. Recommendations for Improvement

-   Compare the model transcription with a manually verified
    ground-truth transcription.
-   Calculate Character Error Rate (CER) and Word Error Rate (WER).
-   Ask the model to mark uncertain words rather than silently guessing.
-   Test with images containing different handwriting styles, blur, low
    contrast, and varied lighting.
-   Repeat the experiment to check transcription consistency.
-   Record GPU memory and utilization, if available, to supplement
    execution-time measurements.

## 9. Conclusion

Experiment 08 produced a structured and readable transcription of a
handwritten letter, identified important details, and generated a
concise summary. The output demonstrates promising basic OCR and
text-understanding capability.

However, exact transcription accuracy has not been independently
measured, and uncertainty reporting was not demonstrated in the returned
output.

**Overall assessment: Promising / functionally successful based on the
returned output, but not quantitatively validated.**

## 10. Source

Source data: `experiment_08_ocr_results.json` supplied for this report.
The evaluation is based on the recorded configuration and model output.
No independent ground-truth transcription was supplied.

Project repository:
https://github.com/abhinash7729/LMM-Vision-Language-Project
