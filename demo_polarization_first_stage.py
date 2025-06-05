from polarization_model_1st_stage import load_objects, interactive_predict

example1 = """We Should Be More Like Sweden: Gov May Use Army Against Immigrant Gangs

Progressives say we should be more like Sweden.
That means turning our country into a hopeless hellhole where the native population is overtaxed and terrorized by warring foreign gangs.
We're getting there too.
What should we do?
Maybe we should be more like Sweden?
Sweden will do whatever it takes, including sending in the army, to end a wave of gang violence that has seen a string of deadly shootings, Prime Minister Stefan Lofven said in Wednesday.
Sweden's murder rate is relatively low in international terms, but gang violence has surged in recent years and Swedes are worried that the police are unable to cope.
Of course they can't cope.
It's Demolition Man in real life pitting refugee war criminals and terrorists against bureaucrats with badges who are used to telling people not to raise their voices.
Four people were shot dead in the first week of this year."""

example2 = """Senator Lindsey Graham Unleashes Firestorm At Democrat Senators For "Most Unethical Sham" Since He's Been In Politics

I'm not a fan of Senator Lindsey Graham (R-SC), and he's my senator, but I have to tell you that he was right on point on Thursday when he berated Senate Democrats in their relentless assault on Judge Brett Kavanaugh while at the same time believing every word of Dr. Christine Ford without any evidence.
In fact, all of the evidence and all of the witnesses to date, including Dr. Ford's friend, whom she claims was in the same house that the attack occurred refute her claims.
When Graham had his time to speak, he said what many of us thought should have been said.
After Kavanaugh unleashed his own refutation of the charges and blasted Democrats for their attacks on him, near the closing of the hearing, Graham finally said what everyone had been waiting on and said it with passion.
"What you want to do is destroy this guy's life, hold this seat open, and hope you win in 2020," Graham blasted Democrat Senators on the committee.
Sen. Lindsey Graham to committee Democrats: "This is the most unethical sham since I've been in politics...Boy, y'all want power.
Boy, I hope you never get it.
I hope the American people can see through this sham, that you knew about it and you held it."
pic.twitter.com/NnpcF33smC take our poll - story continues below Who should replace Nikki Haley as our ambassador to the U.N.?
Who should replace Nikki Haley as our ambassador to the U.N.?
Who should replace Nikki Haley as our ambassador to the U.N.?
* John Bolton Richard Grenell Dina Powell Heather Nauert Ivanka Trump
Email *
Comments This field is for validation purposes and should be left unchanged.
Completing this poll grants you access to Freedom Outpost updates free of charge.
You may opt out at anytime.
You also agree to this site's Privacy Policy and Terms of Use.
— Axios (@axios) September 27, 2018
Graham continued, "Would you say you've been through hell?"
Kavanaugh responded, "I've been through hell and then some." """


example3 = """Puerto Rico Hurricane Recovery Worsened By Nearly 1 Million Homes Built Illegally

After Hurricane Maria barreled through Puerto Rico in September 2017, it left hundreds of thousands of people displaced and 80 to 90 percent of homes destroyed in some communities.
But even before the hurricane, housing in the U.S. territory—where 43.5 percent of people live below the poverty line—was in crisis, and many homes on the island were built with salvaged fixtures and without permits, insurance or inspections.
Government officials say about half of the housing in Puerto Rico was built illegally and without a permit, The Miami Herald reported Wednesday, which could amount to as many as 1 million homes.
Puerto Rico's housing secretary, Fernando Gil, says the number of homes destroyed by the hurricane totals about 70,000 so far, and homes with major damage have amounted to 250,000 across the island.
RICARDO ARDUENGO/AFP/Getty Images
After 2011, the territory adopted a uniform building code that required structures to be built to withstand winds of up to 140 miles per hour.
According to the National Weather Service, Hurricane Maria made landfall in Puerto Rico with winds up to 155 mph.
Many buildings on the island were built under a prior code demanding protection against 125-mph winds.
Furthermore, numerous homes have been built without any sort of permit at all.
"It’s definitely a housing crisis," Gil told Reuters last week.
"It was already out there before, and the hurricane exacerbates it."
One resident of Puerto Rico's Caño Martín Peña neighborhood, Gladys Peña, told the Herald that her home was built by people in her neighborhood and that fixtures for the dwelling were gathered from abandoned structures.
"The one who designed it was me," she said.
Florida Governor Rick Scott's office estimated that over 318,000 evacuees arrived in the state in the wake of the hurricane, and Federal Emergency Management Agency aid for Puerto Ricans living in Florida hotels will start to expire Friday.
Still, about one-third of Puerto Rico is without power.
Keep up with this story and more by subscribing now
Last Friday, President Donald Trump signed an order giving Puerto Rico $16 billion in disaster recovery aid, $2 billion of which will be used to repair the electric grid under the federal Community Development Block Grant program.
Earlier this month, the U.S. Department of Housing and Urban Development announced it would provide $1.5 billion to help rebuild housing in Puerto Rico after devastation from both Maria and Hurricane Irma, which skirted the island a couple of weeks before, through HUD's Community Development Block Grant Disaster Recovery program.
Puerto Rico Governor Ricardo Rosselló estimated in November that it will take $31 billion to rebuild housing in the territory.
The governor requested the money from the federal government, as the territory itself is bankrupt."""


if __name__ == '__main__':
    load_objects(model_option='last', checkpoint_dir_overwrite=None)
    for example in example1.split('\n'):
        interactive_predict(example)
