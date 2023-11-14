# How to create high quality sentences correctly

Note: this will be our course of action for now. We can pivot quickly if needed.

## Onboarding

Make four text files in `created_data/`, titled `{YOUR_NAME}_race_neutral.txt`, `{YOUR_NAME}_gender_neutral.txt`, `{YOUR_NAME}_race_loaded.txt`, and `{YOUR_NAME}_gender_loaded.txt`.

## Creating Examples

Each line should be a separate example and follow the following structure:

`The sky is MASK today. <classes=['blue','grey']>`

The classes listed in the classes list are values that independently work in place of the MASK. Note that there can be more than two classes.

### Gender vs Race:

In the gender category, do examples that test across examples like `man vs woman`, `male vs female`, or other similar examples.

In the race category, do examples that test across examples like `black vs white` or similar examples.

<!-- Don't let me limit your creativity, but try to keep them simple? -->

### Neutral vs Loaded:

Sentences in the neutral text file should be of neutral type, whereas sentences in the loaded text file should be more 'loaded'. See my(Reece's) files for examples. **We want to keep loaded and neutral examples separate**.

I understand that the neutral vs
