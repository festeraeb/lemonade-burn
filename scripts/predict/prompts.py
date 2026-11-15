# Prompt sets for the predictor study (docs/ENGINE_PREDICTOR.md).
# TRAIN / VAL: new prompts, used to fit and tune predictors. TEST_A / TEST_B: the cm.py sets used for the
# final measurements; never used for fitting or tuning.
TRAIN = [
 "Write a Python function that merges overlapping intervals and explain its complexity.",
 "Give me a recipe for vegetarian chili and explain why each spice is used.",
 "Write a short poem about the sea at night, then explain the imagery you chose.",
 "Explain how the Roman Republic became an empire, in detail.",
 "Translate the following into French and explain two tricky grammar points: 'The old library closes early on Sundays because the volunteers go to the market.'",
 "Solve step by step: a train leaves at 3pm at 80 km/h, a second leaves the same station at 4pm at 100 km/h. When does the second catch up?",
 "Write a SQL schema for a bookstore with authors, books and orders, and a query for the top five authors by revenue.",
 "Explain the difference between TCP and UDP and when you would choose each.",
 "Write a friendly email to a landlord asking to fix a leaking kitchen tap, with a polite deadline.",
 "What are the main arguments for and against a universal basic income? Be balanced.",
 "Describe how vaccines train the immune system, with the cell types involved.",
 "Write a bash script that backs up a directory to a dated tar file and keeps only the last seven.",
 "Give a three-day travel itinerary for Lisbon with food suggestions and transport tips.",
 "Explain compound interest with a worked example and a common mistake people make.",
 "Tell me a story about a lighthouse keeper who finds a message in a bottle.",
 "Explain how a transformer language model works to a curious teenager.",
 "Write a Rust function that parses a comma separated list of integers and returns the sum, with error handling.",
 "Describe the rules of chess for a complete beginner, including castling and en passant.",
 "Explain what causes the seasons on Earth and why a common explanation is wrong.",
 "Write a dialogue between a skeptical customer and a salesperson selling an electric bicycle.",
 "Summarise the plot of Hamlet and discuss its main themes.",
 "Explain how to read a balance sheet and what the key ratios mean.",
 "Give me a beginner weekly strength training plan and explain the reasoning for rest days.",
 "Explain how the Linux kernel schedules processes, in a clear structured way.",
 "Write a JSON document describing a fictional library with three books, then explain each field.",
 "Explain music theory basics: scales, chords and why a minor key sounds sad.",
 "Discuss the causes of the 2008 financial crisis in plain language.",
 "Explain the difference between weather and climate and how climate models are tested.",
 "Give a detailed guide to growing tomatoes on a small balcony.",
 "Write a haiku sequence about autumn and then a brief commentary.",
]
VAL = [
 "Write a JavaScript function that debounces another function and explain closures.",
 "Explain how bread rises and why kneading matters.",
 "What happened during the Cuban Missile Crisis and how was it resolved?",
 "Write a cover letter for a junior data analyst position.",
 "Explain how GPS determines your position, including the role of atomic clocks.",
 "Describe the water cycle and how human activity affects it.",
]
TEST_A = [
 "Write a long, detailed essay about the history of sonar and how side-scan sonar is used to find shipwrecks.",
 "Write a Rust program that reads a CSV file with quoted fields, groups rows by the first column and prints per-group totals, with explanations.",
 "Explain in detail the causes and consequences of the 1913 Great Lakes storm.",
 "Describe step by step how a magnetometer survey is planned, run and interpreted to find a steel wreck.",
 "Explain what a mixture-of-experts language model is, how routing works, and why it is cheaper to run than a dense model.",
]
TEST_B = [
 "Give a detailed recipe for a traditional beef stew and explain each technique used, including how to adjust seasoning.",
 "Explain how a binary search tree works, then write a Python class implementing insert, search and delete with comments.",
 "Write a detailed summary of the main causes of the French Revolution and how they connected to one another.",
 "Explain the differences between common law and civil law legal systems with several concrete examples of each.",
 "Describe how photosynthesis works at the molecular level, including the light reactions and the Calvin cycle.",
]
