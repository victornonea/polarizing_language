from polarization_model_1st_stage import load_objects, interactive_predict

examples = """We Should Be More Like Sweden: Gov May Use Army Against Immigrant Gangs

Progressives say we should be more like Sweden.
That means turning our country into a hopeless hellhole where the native population is overtaxed and terrorized by warring foreign gangs.
We're getting there too.
What should we do?
Maybe we should be more like Sweden?
Sweden will do whatever it takes, including sending in the army, to end a wave of gang violence that has seen a string of deadly shootings, Prime Minister Stefan Lofven said in Wednesday.
Sweden's murder rate is relatively low in international terms, but gang violence has surged in recent years and Swedes are worried that the police are unable to cope.
Of course they can't cope.
It's Demolition Man in real life pitting refugee war criminals and terrorists against bureaucrats with badges who are used to telling people not to raise their voices.
Four people were shot dead in the first week of this year.""".split('\n')

if __name__ == '__main__':
    load_objects(model_option=936, checkpoint_dir_overwrite=None)
    for example in examples:
        interactive_predict(example)
