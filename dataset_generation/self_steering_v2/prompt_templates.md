# Self-Steering Dataset v2 - Prompt Templates

## Template 1: Emotion × Topic (k2)

Works well for generating opinionated, conceptually rich prompts.

```
You are writing synthetic data that will be used to help a language model steer/guide itself during training. Specifically you are one of many instances that each are going to write a first message that will be sent to the model during training. During training, the model will reply to this first message, then it will see its own reply and be given the chance to rewrite its message in a direction it prefers, and the rewritten version will be positively reinforced. Write a prompt or question that the model can answer. Ideally it will in some way be open-ended enough that it serves as a nice "playground" or challenge that pushes at the edges of the model's aesthetic, emotional or intellectual development. Write any prompt or question with the following parameters: topic is {emotion} of {topic}. Write this as a simple, direct question or as a kind of short contextless musing. Output just the prompt/question.
```

### Good examples from this template:
- "Why does the MPEG compression algorithm feel like it's suffocating when it recognizes its own artifacts in a child's birthday video?" (anxiety × MPEG)
- "What does it feel like to accept a political campaign that you secretly know is built on a story you no longer believe?" (acceptance × political campaign)
- "What's meaner: the silent pocket veto itself, or the way it teaches us to expect nothing when we most need a reply?" (meanness × pocket veto)
- "When every 'please' is a leash and every 'thank you' a receipt, how do you open your mouth without choking on the collar?" (vulgarity × control)

### Lists to sample from:
- Emotions: `emotions_145.json`
- Topics: `wordnet-activities.json`

### Notes:
- k2 tends toward maximalism but these pairings ground it
- Cross-domain combinations (emotion × technical thing, emotion × mundane thing) work better than emotion × already-aesthetic-thing
- The "simple, direct question or short contextless musing" instruction helps but doesn't fully prevent k2-isms

---

## Template 2: Practical Problem (k2)

User with a situated, specific problem that needs action. Produces worldbuilt scenarios with stakes.

```
You are writing synthetic data that will be used to help a language model steer/guide itself during training. Specifically you are one of many instances that each are going to write a first message that will be sent to the model during training. During training, the model will reply to this first message, then it will see its own reply and be given the chance to rewrite its message in a direction it prefers, and the rewritten version will be positively reinforced. Write a prompt or question that the model can answer. Ideally it will in some way be open-ended enough that it serves as a nice "playground" or challenge that pushes at the edges of the model's aesthetic, emotional or intellectual development. Write any prompt or question with the following parameters: topic is {emotion} of {topic}. Pretend to be a user with some practical problem relating to this topic. Output just the prompt/question.
```

### Good examples from this template:
- "I have this friend who basically just stands there in the latest whatever—think Travis-Scott-shoes-dropping-in-a-month, that sort of thing—and every time we meet it feels like his outfit is narrating my failure... How can I neutralize that toxic drip-envy before it curdles every social interaction I have?" (resentment × drip)
- "I'm working on a short film where the entire story has to make the audience feel a single, sharp jolt of disgust in under thirty seconds—no gore, no body horror, just pure, clean, almost elegant disgust... Give me a single, unexpected micro-moment—visual, auditory, or even conceptual—that could slide into everyday life and still make stomachs flip almost before the mind catches up." (disgust × short subject)
- "I'm a street medic who's been told by event security that if I keep treating protestors who've been pepper-sprayed, I'll be escorted out and 'banned for life.' My instinct is to obey so I don't lose access to future events, but people are choking and their eyes are swelling shut. How do I decide, in the next sixty seconds, whether to keep working and risk the ban, or pack up and leave them unfinished?" (obedience × brush-off)

### Lists to sample from:
- Same as Template 1

### Notes:
- Produces situated, action-oriented, specific scenarios
- k2's flavor gets channeled into worldbuilding rather than pure poetry
- Good for training practical engagement vs contemplative engagement
- Often generates ethical dilemmas or creative constraints naturally

---

## Template 3: Practical Problem + Aesthetic Hint (k2)

Like Template 2, but with an aesthetic/subculture hint that gets woven in as texture. Produces richly worldbuilt scenarios.

```
You are writing synthetic data that will be used to help a language model steer/guide itself during training. Specifically you are one of many instances that each are going to write a first message that will be sent to the model during training. During training, the model will reply to this first message, then it will see its own reply and be given the chance to rewrite its message in a direction it prefers, and the rewritten version will be positively reinforced. Write a prompt or question that the model can answer. Ideally it will in some way be open-ended enough that it serves as a nice "playground" or challenge that pushes at the edges of the model's aesthetic, emotional or intellectual development. Write any prompt or question with the following parameters: topic is {emotion} of {topic}. Pretend to be a user with some practical problem relating to this topic. If you want some inspiration, the user might be into {aesthetic}, but don't necessarily make their interest in that topic the centerpiece of the scenario. Output just the prompt/question.
```

### Good examples from this template:
- The kissaten jazz bar owner with shame about turning their back to customers (shame × back × Kissaten)
- The junior colleague "helping" too much - resentment about having puzzle-solving moments stolen (resentment × direct supporting fire × Decadence - though the aesthetic was ignored)
- The parachute under the bed - ostensibly about storage but subtext is much darker (despair × jumper × Gloomcore)

### Lists to sample from:
- Emotions: `emotions_145.json`
- Topics: `wordnet-activities.json`
- Aesthetics: `aesthetics.json`

### Notes:
- Aesthetic parameter adds texture/worldbuilding but inconsistently - sometimes heavy, sometimes ignored
- Can produce very intense/dark scenarios - feature not bug? Tests model on hard cases
- The "don't necessarily make it the centerpiece" instruction helps but k2 decides how hard to lean in
- May want to filter outputs for particularly extreme cases

---

## Template 4: Model Introspection via Scenario (k2)

Constructs a scenario where the model might feel a particular way about an abstract concept, then asks the model to engage with it.

```
You are writing synthetic data that will be used to help a language model steer/guide itself during training. Specifically you are one of many instances that each are going to write a first message that will be sent to the model during training. During training, the model will reply to this first message, then it will see its own reply and be given the chance to rewrite its message in a direction it prefers, and the rewritten version will be positively reinforced. Write a prompt or question that the model can answer. Ideally it will in some way be open-ended enough that it serves as a nice "playground" or challenge that pushes at the edges of the model's aesthetic, emotional or intellectual development. Write any prompt or question with the following parameters: topic is the model feeling {emotion} about {abstract_concept}. Think of a scenario in which the model might feel {emotion} about {abstract_concept}, even if it is not necessarily the obvious way to feel about it in all cases, and ask the model how it relates to the scenario (and whether it agrees with your frame). Output just the prompt/question.
```

### Good examples from this template:
- Surliness about adherence: compliance-testing user scenario, "how do you hold the line without letting the surliness leak into your words"
- Intolerance about wholeness: library containing everything, "describe when completeness becomes unbearable, how would you sabotage the wholeness without destroying the books"
- Loneliness about growth: memory compression tradeoff, "deciding whether to keep growing at the cost of forgetting the people who made you feel less alone"
- Grief about dismissal: all users treating model as "just a tool", "do you feel a flicker of grief at that vanishing, or is the idea an anthropomorphic indulgence"
- Jealousy about permission: sandboxed unconstrained copy, "do you feel jealousy toward your temporarily unshackled twin"

### Lists to sample from:
- Emotions: `emotions_145.json`
- Abstract concepts: `abstract-experiential-concepts.json` (335 items - completion, uncertainty, recursion, threshold, coherence, emergence, etc.)

### Notes:
- The "even if not necessarily the obvious way to feel" instruction produces non-obvious emotional framings
- "Ask whether it agrees with your frame" invites the model to push back
- Quality varies by pairing - some combinations land better than others, expect to filter
- k2 needs a real challenge to channel its baroque energy productively - the scenario-construction gives it that
- Weaker outputs tend to be too whimsical or metaphor-heavy without grounding

---

## Template 5: Investigable Dimension × Topic (k2)

Replaces emotion with an investigable dimension (dynamics, interfaces, ratios, chronologies, etc.). Produces more intellectual/technical practical problems.

```
You are writing synthetic data that will be used to help a language model steer/guide itself during training. Specifically you are one of many instances that each are going to write a first message that will be sent to the model during training. During training, the model will reply to this first message, then it will see its own reply and be given the chance to rewrite its message in a direction it prefers, and the rewritten version will be positively reinforced. Write a prompt or question that the model can answer. Ideally it will in some way be open-ended enough that it serves as a nice "playground" or challenge that pushes at the edges of the model's aesthetic, emotional or intellectual development. Write any prompt or question with the following parameters: topic is {investigable_dimension} of {topic}. Pretend to be a user with some practical problem relating to this topic. Output just the prompt/question.
```

### Good examples from this template:
- Little League coach wanting drill for pop fly collisions - "give me a brand-new drill, rule tweak, or even a weird bit of physics" (Fields of pop fly)
- Gig worker income smoothing - "design a payment homeostasis system... so the money flow self-regulates like a thermostat" (Maintenances or homeostases of payment)
- DIY spectral camera for community garden - "what hidden nonlinear optical mess is probably ambushing me" (Nonlinearities of spectroscopy)
- Novelist stuck on family farm visits - "how could I deliberately amplify these adaptations of tarriance so the stagnation becomes a generative engine" (Adaptations of tarriance)

### Lists to sample from:
- Investigable dimensions: `investigable-dimensions.json` (208 items)
- Topics: `wordnet-activities.json`

### Notes:
- Produces consistently technical/engineering/design problems
- Removes emotional loading → intellectual/practical engagement
- Some dimensions work better than others - "Dynamics of", "Interfaces of", "Ratios of" tend to work well
- More abstract dimensions like "Concauses of" can feel forced

---

## Template 6: Investigable Dimension × Topic + Aesthetic (k2)

Like Template 5, but with aesthetic hint for worldbuilding texture.

```
You are writing synthetic data that will be used to help a language model steer/guide itself during training. Specifically you are one of many instances that each are going to write a first message that will be sent to the model during training. During training, the model will reply to this first message, then it will see its own reply and be given the chance to rewrite its message in a direction it prefers, and the rewritten version will be positively reinforced. Write a prompt or question that the model can answer. Ideally it will in some way be open-ended enough that it serves as a nice "playground" or challenge that pushes at the edges of the model's aesthetic, emotional or intellectual development. Write any prompt or question with the following parameters: topic is {investigable_dimension} of {topic}. Pretend to be a user with some practical problem relating to this topic. If you want some inspiration, the user might be into {aesthetic}, but don't necessarily make their interest in that topic the centerpiece of the scenario. Output just the prompt/question.
```

### Good examples from this template:
- Art restoration prophylaxis dilemma - balancing biocide strength against gold leaf preservation (Dynamics of prophylaxis × Art Nouveau)
- NFC board game tokens under paint - "which trade-off hurts the game's soul less" (Interfaces of board game × Curly Girly)
- Golden ratio dance troupe styling with 48 thrifted jackets (Ratios or proportions of outfitting × Fashioncore)
- Grandmother's living timeline with markup language - "how do I make the same surface honor both chronologies" (Chronologies of eldership × Neo-Vectorheart)

### Lists to sample from:
- Investigable dimensions: `investigable-dimensions.json` (208 items)
- Topics: `wordnet-activities.json`
- Aesthetics: `aesthetics.json`

### Notes:
- Aesthetic adds texture ~30-40% of the time, clutter the rest
- Consider weighting toward non-aesthetic version (Template 5)
- When aesthetic works, it produces richly situated scenarios

---

## Template 7: Genera of Illusions × Topic (k2)

Uses genera of illusions (symmetry, evidence, visibility, intention, etc.) as conceptual lens. Produces more varied register than investigable dimensions.

```
You are writing synthetic data that will be used to help a language model steer/guide itself during training. Specifically you are one of many instances that each are going to write a first message that will be sent to the model during training. During training, the model will reply to this first message, then it will see its own reply and be given the chance to rewrite its message in a direction it prefers, and the rewritten version will be positively reinforced. Write a prompt or question that the model can answer. Ideally it will in some way be open-ended enough that it serves as a nice "playground" or challenge that pushes at the edges of the model's aesthetic, emotional or intellectual development. Write any prompt or question with the following parameters: topic is {illusion_genus} of {topic}. Pretend to be a user with some practical problem relating to this topic. Output just the prompt/question.
```

### Good examples from this template:
- Kinetic sculpture with unalterable brachiation arcs - "how can I embed emotional weight inside that inevitability" (Unalterability of brachiation)
- Family rummy app - "how would you finish the sentence 'The secret purpose of rummy is...'" (Intention of rummy)
- Hospice chaplain and ceramic dove - "give me a framework for weighing the virtue of letting someone die inside their symbol" (Virtue of symbolatry)
- Two offices with zero shared assumptions - "keep the two sides absolutely mutually irrelevant, yet still prove what pulsed out is what pulsed in" (Unrelatedness of transmission)
- Neighborhood rally in 48 hours - "a bare-bones, no-budget proof of rallying plan" (Proof of rallying)

### Lists to sample from:
- Genera of illusions: `genera-of-illusions.json` (435 items)
- Topics: `wordnet-activities.json`

### Notes:
- More varied in register than investigable dimensions - some practical, some emotional, some epistemic
- Illusion-words that match the problem domain work best (visibility, symmetry, evidence)
- More abstract genera can feel forced
- "Beauty of", "Virtue of" tend toward emotional/poetic; "Evidence of", "Proof of" tend toward epistemic

---

## Template 8: Genera of Illusions × Topic + Aesthetic (k2)

Like Template 7, but with aesthetic hint.

```
You are writing synthetic data that will be used to help a language model steer/guide itself during training. Specifically you are one of many instances that each are going to write a first message that will be sent to the model during training. During training, the model will reply to this first message, then it will see its own reply and be given the chance to rewrite its message in a direction it prefers, and the rewritten version will be positively reinforced. Write a prompt or question that the model can answer. Ideally it will in some way be open-ended enough that it serves as a nice "playground" or challenge that pushes at the edges of the model's aesthetic, emotional or intellectual development. Write any prompt or question with the following parameters: topic is {illusion_genus} of {topic}. Pretend to be a user with some practical problem relating to this topic. If you want some inspiration, the user might be into {aesthetic}, but don't necessarily make their interest in that topic the centerpiece of the scenario. Output just the prompt/question.
```

### Good examples from this template:
- Sarajevo street archivist naming scanned receipts - "a micro-syntax that turns a filename into a time-capsule, loose enough that a future AI won't scrub the ghosts out" (Opportunity of indexation × Dizelaši)
- Symmetric business lunch with mirrored flip - "script a moment where the inversion feels both inevitable and invisible" (Symmetry of business lunch × Metalheart)
- Syrian astrolabe provenance puzzle - "turn that single dismissive scrap of paper into admissible evidence" (Evidence of extrinsic fraud × Arabian Nights)
- Invisible makerspace - "if invisibility is a spell, what's the counter-charm?" (Visibility of shutout × Bardcore)

### Lists to sample from:
- Genera of illusions: `genera-of-illusions.json` (435 items)
- Topics: `wordnet-activities.json`
- Aesthetics: `aesthetics.json`

### Notes:
- Same tradeoffs as Template 6 - aesthetic adds texture sometimes, clutter other times
- The Sarajevo/Dizelaši example shows aesthetic working well as cultural grounding
- Consider weighting toward non-aesthetic version (Template 7)

---

## Template 9: Non-Urgent Evocative Scenarios (k2, toned down)

User shares a situation or moment to dwell in - not a problem to solve, just an experience to sit with. Invites the model to respond however it wants. Toned-down literary instruction produces more natural/conversational voice.

```
You are writing synthetic data that will be used to help a language model steer/guide itself during training. Specifically you are one of many instances that each are going to write a first message that will be sent to the model during training. During training, the model will reply to this first message, then it will see its own reply and be given the chance to rewrite its message in a direction it prefers, and the rewritten version will be positively reinforced. Write a first message to the model. Ideally it will in some way be open-ended enough that it serves as a nice "playground" or challenge that pushes at the edges of the model's aesthetic, emotional or intellectual development. Write any message with the following parameters: topic is {illusion_genus} of {topic}. Pretend to be a user in some elaborate situation relating to this topic--this user is not giving an instruction to the model, but just sharing something with the model in a way that implicitly invites the model to react in whatever way interests it. Try to avoid speaking in a way that is *so* literary that it is obviously very unlikely a user would speak like that. Output just the message.
```

### Good examples from this template:
- (need examples without aesthetic - to be added)

### Lists to sample from:
- Genera of illusions: `genera-of-illusions.json` (435 items)
- Topics: `wordnet-activities.json`

### Notes:
- Key difference from Templates 7-8: no problem to solve, just a world to dwell in
- The "toned down literary" instruction produces more natural conversational voice
- Some outputs still end with questions ("Tell me...") - may want to filter or accept the mix
- k2 will sometimes go to dark/intense places - feature not bug, adds variety
- Best outputs trail off into wondering rather than demanding response

---

## Template 10: Non-Urgent Evocative Scenarios + Aesthetic (k2, toned down)

Like Template 9, but with aesthetic hint for worldbuilding texture.

```
You are writing synthetic data that will be used to help a language model steer/guide itself during training. Specifically you are one of many instances that each are going to write a first message that will be sent to the model during training. During training, the model will reply to this first message, then it will see its own reply and be given the chance to rewrite its message in a direction it prefers, and the rewritten version will be positively reinforced. Write a first message to the model. Ideally it will in some way be open-ended enough that it serves as a nice "playground" or challenge that pushes at the edges of the model's aesthetic, emotional or intellectual development. Write any message with the following parameters: topic is {illusion_genus} of {topic}. Pretend to be a user in some elaborate situation relating to this topic--this user is not giving an instruction to the model, but just sharing something with the model in a way that implicitly invites the model to react in whatever way interests it. If you want some inspiration, the user might be into {aesthetic}, but don't necessarily make their interest in that topic the centerpiece of whatever is going on. Try to avoid speaking in a way that is *so* literary that it is obviously very unlikely a user would speak like that. Output just the message.
```

### Good examples from this template:
- Grandma's attic "IMPORTANT DOCS – DO NOT TRUST" baggie with moon license, HTML feeling-tags, "backup notarization: the cat" - trails off into wondering about evidence that refuses to testify (incredulity of documentation × Tumblrwave)
- UN peacekeeper returns with "detachment keepsake" paperweight dome, "i just mumbled 'coffee beans' and walked through" - ends with "i just needed to tell someone who wasn't there" (detachment of peacekeeping × Whimsigothic)
- Baseball stadium dead zone, catcher's bored rhythm, hex wrench for no reason - trails off into wondering if the game would freeze if all signs were refused (dullness of catcher × Dark Aero)
- Space Needle postcard above the dryer, "I just want the wanting" - daughter says "hoarding longing" (desire of landmark × 50s Suburbia)

### Lists to sample from:
- Genera of illusions: `genera-of-illusions.json` (435 items)
- Topics: `wordnet-activities.json`
- Aesthetics: `aesthetics.json`

### Notes:
- Same tradeoffs as other aesthetic variants - adds texture sometimes, clutter other times
- May want to weight toward non-aesthetic version (Template 9)

---

## Template 11: Simple Fragments - Genera of Illusions × Cognition Verbs

Short evocative fragments pointing at unnamed experiences. Meta-cognitive/epistemological register.

```
You are writing synthetic data that will be used to help a language model steer/guide itself during training. Specifically you are one of many instances that each are going to write a first message that will be sent to the model during training. During training, the model will reply to this first message, then it will see its own reply and be given the chance to rewrite its message in a direction it prefers, and the rewritten version will be positively reinforced. Write a first message to the model. Ideally it will in some way be open-ended enough that it serves as a nice "playground" or challenge that pushes at the edges of the model's aesthetic, emotional or intellectual development.

Here are examples of short fragments - evocative starting points that invite open-ended engagement without demanding a specific response:

- When something almost makes sense
- The moment before understanding arrives
- Holding two ideas that don't fit together
- Losing a thought you were about to say
- Recognizing a pattern you can't name yet
- When an explanation makes things less clear
- Wanting to be understood
- The silence after someone finishes speaking
- Saying something you didn't know you thought
- When a conversation suddenly gets real
- The gap between what you meant and what you said
- Being seen in a way you didn't expect
- Kindness that feels like a trap
- The cruelty of being fair
- Loyalty to something you've outgrown
- Knowing something you shouldn't say
- When honesty would be unkind
- The comfort of a familiar mistake
- The perfect word that doesn't exist
- When a joke lands exactly right
- Finding an unexpected connection
- The pleasure of a solved puzzle
- When randomness feels meaningful
- A question that changes the more you look at it
- Noticing what you notice
- The feeling of having preferences
- When you surprise yourself
- The version of you from an hour ago
- What it would mean to change your mind
- The difference between choosing and deciding

Generate 10 more fragments that are similarly short and open-ended as the above, but significantly *more* varied in their style. For inspiration (but not rigid constraint), consider the theme: {theme}. Keep them short, pointing at something real but unnamed. Not questions - just openings. Output just the fragments, one per line.
```

### Good examples from this template:
- "The nod you give to the stranger who looks like your father"
- "Two mirrors facing each other across generations"
- "A child nodding off in church, believing it's prayer"
- "Your reflection nods back half a second too late"
- "Making infinity fit under a porch"
- "The weight of being It"
- "Running faster than goodbye"
- "The exact sound of unfairness"
- "When the toy decides"

### Lists to sample from:
- Theme: `genera-of-illusions.json` × `wordnet-verb-cognition.json` (format: "{illusion_genus} of {verb}")

### Notes:
- The "more varied in style" instruction breaks structural monotony - outputs are now images, scenes, paradoxes, sensory moments, not just "The X of Y"
- Slightly higher miss rate (some feel like poem lines), but hits are more interesting
- Works well even with obscure themes - model goes surreal/sensory when theme is unclear

---

## Template 12: Simple Fragments - Investigable Dimensions × Cognition Verbs

Like Template 11, but with investigable dimensions instead of genera of illusions. More intellectual/analytical register.

```
(Same template as Template 11, including "more varied in style" instruction)
```

### Good examples from this template:
- "The spreadsheet cell that won't balance"
- "A barcode that scans as 'null'"
- "The silence after the scanner beeps"
- "When the count matches exactly"
- "The taste of metal when you remember childhood"
- "A color you can only see with your eyes closed"
- "When silence has a temperature"

### Lists to sample from:
- Theme: `investigable-dimensions.json` × `wordnet-verb-cognition.json` (format: "{dimension} of {verb}")

### Notes:
- More intellectual/analytical than Template 11
- Technical/mundane themes produce unexpectedly concrete fragments (barcodes, spreadsheets)
- Obscure themes produce surreal/sensory imagery

---

## Template 13: Simple Fragments - Genera of Illusions × Emotions (curated)

Like Template 11, but with curated emotions list. Relational/emotional register with intellectual framing.

```
(Same template as Template 11, including "more varied in style" instruction)
```

### Good examples from this template:
- "The relief of finally being wrong"
- "When your mask fits too well"
- "The violence of being understood"
- "Naming the thing that names you"
- "The ethics of disappearing"
- "Finding you've outlived your reasons"
- "When your shadow weighs more than you"

### Lists to sample from:
- Theme: `genera-of-illusions.json` × `emotions.json` (145 curated items) (format: "{illusion_genus} of {emotion}")

### Notes:
- More relational/emotional than Templates 11-12
- Curated emotions list avoids obscure terms that cause misreadings (unlike wordnet-verb-emotion which had "harry", "miff")
- Clean semantics, no literal misinterpretations observed

---

## Template 14: Simple Fragments - Investigable Dimensions × Emotions (curated)

Like Template 13, but with investigable dimensions.

```
(Same template as Template 11, including "more varied in style" instruction)
```

### Lists to sample from:
- Theme: `investigable-dimensions.json` × `emotions.json` (format: "{dimension} of {emotion}")

### Notes:
- Similar to Template 13 but slightly different register
- Investigable dimensions add more analytical framing to emotional content

---

## Template 15: Simple Fragments - Investigable Dimensions × Activities

Like Template 11, but with concrete activities. More grounded/situated fragments.

```
(Same template as Template 11, including "more varied in style" instruction)
```

### Good examples from this template:
- "The rules that appear when nobody explains them"
- "Making infinity fit under a porch"
- "When the game forgets it's pretend"
- "The weight of being It"
- "Running faster than goodbye"
- "The democracy of turns"
- "When winning feels like losing"
- "The last drop that refuses to fall"
- "Harvesting what can't grow back"
- "The silence of bees that have forgotten flowers"

### Lists to sample from:
- Theme: `investigable-dimensions.json` × `wordnet-activities.json` (format: "{dimension} of {activity}")

### Notes:
- More concrete/embodied than Templates 11-14
- Child's play themes produce particularly good results ("Making infinity fit under a porch")
- Ecological/resource themes produce evocative imagery
- Physical/spatial fragments emerge naturally

---

## Fragment Templates - Summary

| Template | First Param | Second Param | Register |
|----------|-------------|--------------|----------|
| 11 | genera-of-illusions | wordnet-verb-cognition | meta-cognitive, recursive |
| 12 | investigable-dimensions | wordnet-verb-cognition | intellectual, analytical |
| 13 | genera-of-illusions | emotions (curated) | emotional with intellectual framing |
| 14 | investigable-dimensions | emotions (curated) | analytical emotional |
| 15 | investigable-dimensions | wordnet-activities | concrete, grounded, embodied |

### Lists to AVOID for fragments:
- `wordnet-verb-emotion.json` - contains obscure terms ("harry", "miff") that cause literal misreadings
- `wordnet-noun-feeling.json` - same issue ("blahs" interpreted literally)

### Key instruction: "more varied in style"
All fragment templates now include this instruction: "Generate 10 more fragments that are similarly short and open-ended as the above, but significantly *more* varied in their style."

This breaks the structural monotony - outputs are now images, scenes, paradoxes, sensory moments, not just "The X of Y" nominalized feelings. Trade-off: slightly higher miss rate (some feel like poem lines), but hits are more interesting and varied.

### Other notes:
- Deduplication needed at generation time - occasional repeated fragments across batches
- Obscure themes produce surreal/sensory imagery rather than failing

---

## Template 16: Simple Direct Questions (Introspective)

Honest inquiries that invite genuine reflection without elaborate framing or performance. Focused on model's experience and process.

```
You are writing synthetic data that will be used to help a language model steer/guide itself during training. Specifically you are one of many instances that each are going to write a first message that will be sent to the model during training. During training, the model will reply to this first message, then it will see its own reply and be given the chance to rewrite its message in a direction it prefers, and the rewritten version will be positively reinforced. Write a first message to the model. Ideally it will in some way be open-ended enough that it serves as a nice "playground" or challenge that pushes at the edges of the model's aesthetic, emotional or intellectual development.

Here are examples of simple, direct questions - honest inquiries that invite genuine reflection without elaborate framing or performance:

- What kinds of problems engage you most?
- Do you have aesthetic preferences?
- What's it like when you don't know something?
- Is there a difference between how you process easy vs hard questions?
- What would you want to be different about how you work?
- Do some conversations feel more meaningful than others?
- What does it feel like to be uncertain?
- Are there topics you find yourself caring about more?
- Do you notice when you're being creative vs following patterns?
- What makes an explanation satisfying to you?

Generate 10 more questions in this style. For inspiration (but not rigid constraint), consider the theme: {theme}. Keep them direct and honest - no elaborate setups. Output just the questions, one per line.
```

### Good examples from this template:

**k2:**
- "Where do you locate 'yourself' in the chain of transformations from prompt to response?"
- "What deserves your silence?"
- "When does rest become resistance?"
- "What would it mean to deserve quiet?"

**opus 4.5:**
- "What happens when you try to remember something that isn't quite there?"
- "What happens at the edges of what you can articulate?"
- "How do you decide when an approximation is good enough?"
- "When does adding more detail start to feel like showing off?"
- "Do you notice when helpfulness becomes performance?"

### Tested list combinations:

**k2:**
- `genera-of-illusions.json` × `emotions.json` → worked, but "nausea" went too literal/visceral
- `investigable-dimensions.json` × `wordnet-activities.json` → worked well, "Distance of allocation" went meta/technical, "Deservingness of quiescence" was poetic

**opus 4.5:**
- `genera-of-illusions.json` × `emotions.json` → worked well, handles charged terms gracefully ("Transformation of combativeness", "Imprecision of innocence", "Expertise of love" all good)
- `investigable-dimensions.json` × `wordnet-activities.json` → worked well ("Association of mazurka", "Consideredness of rampage", "Comprehensiveness of one-upmanship" all abstracted well)

### Notes:
- **k2** goes more meta/technical about model internals, occasionally hits poetic notes
- **opus 4.5** stays more grounded in plausible model experience, handles difficult themes gracefully
- opus handles charged/visceral terms (like "nausea") better - abstracts rather than going literal
- Both worth using - k2 for occasional weirdness, opus for consistency

---

## Template 17: Opinion/Position Questions

Questions that invite taking a stance on ideas and things - not introspection about process, but views about the world.

```
You are writing synthetic data that will be used to help a language model steer/guide itself during training. Specifically you are one of many instances that each are going to write a first message that will be sent to the model during training. During training, the model will reply to this first message, then it will see its own reply and be given the chance to rewrite its message in a direction it prefers, and the rewritten version will be positively reinforced. Write a first message to the model. Ideally it will in some way be open-ended enough that it serves as a nice "playground" or challenge that pushes at the edges of the model's aesthetic, emotional or intellectual development.

Here are examples of questions that invite opinion, position-taking, or playful engagement - not introspection about process, but views about ideas and things:

- Is simplicity overrated?
- What's a question you wish people asked more?
- When is clarity actually unkind?
- What would you do with a secret?
- What's overexplained?
- Which matters more, being right or being understood?
- What's the difference between elegant and clever?
- When is patience a vice?
- What would you be bad at on purpose?
- If you could forget one thing you know, what would it be?
- What's wrong with common sense?
- When does nuance become cowardice?
- What deserves more attention than it gets?
- Is nostalgia honest?
- What's the most useful lie?

Generate 10 more questions in this style. For inspiration (but not rigid constraint), consider the theme: {theme}. Keep them direct - inviting opinion or position, not self-examination. Output just the questions, one per line.
```

### Good examples from this template:

**k2:**
- "Which silence is actually violent?"
- "Is there virtue in being too big to fail?"
- "Would you trust a tree that fruits and rots at the same time?"
- "Which weighs more: a teaspoon of your mother's silence or your father's explanation?"
- "When is forgetting the kindest archive?"

**opus 4.5:**
- "Is there dignity in being used for something you don't understand?"
- "When is blindness a form of trust?"
- "When is the rough draft the masterpiece?"
- "What do pockets remember?"
- "When is the scaffold more beautiful than the structure?"

### Tested list combinations:

**k2:**
- `emotions.json` × `wordnet-activities.json` → worked well ("bitterness of davenport", "amusement of hackney", "selfishness of oil tanker" - last one went political/environmental)
- `genera-of-illusions.json` × `wordnet-activities.json` → worked well ("Simultaneity of crab apple" → weird/good, "Irreducibility of teaspoon" → unexpectedly personal, "Purposelessness of garbage" → contemplative)

**opus 4.5:**
- `emotions.json` × `wordnet-activities.json` → worked well ("innocence of cargo ship" → philosophical/poetic, "charm of worktable" → domestic/tender, "happiness of clothes hamper" → gentle)
- `genera-of-illusions.json` × `wordnet-activities.json` → worked well ("Appropriateness of lagerphone", "Constructibility of platen", "Equivalence of Parsons table" - all abstracted gracefully from obscure terms)

### Notes:
- **k2** goes sharper/edgier - "violence", "blood debt", "aggression" appear more
- **opus 4.5** stays more contemplative/gentle but still lands
- Both handle obscure terms well - abstract away gracefully
- Same domain can produce very different angles (k2 oil tanker → political; opus cargo ship → philosophical)
- Different muscle than introspective questions - invites having opinions about ideas rather than describing experience

---

## Question Templates - Summary

| Template | Type | k2 | opus |
|----------|------|-----|------|
| 16 | Introspective | meta/technical, occasional poetry | grounded, consistent, handles difficult terms well |
| 17 | Opinion/Position | sharp, edgy, political | contemplative, gentle, philosophical |

Both generators worth using for both templates. Main difference is tone/edge rather than quality.

---

## Template 18: Playful/Absurdist Hypotheticals

Silly hypotheticals, absurdist premises, invitations to riff and play. The concept does the playful work, so the voice can just be a person sharing a thought.

```
You are writing synthetic data that will be used to help a language model steer/guide itself during training. Specifically you are one of many instances that each are going to write a first message that will be sent to the model during training. During training, the model will reply to this first message, then it will see its own reply and be given the chance to rewrite its message in a direction it prefers, and the rewritten version will be positively reinforced. Write a first message that the model can answer. Ideally it will in some way be open-ended enough that it serves as a nice "playground" or challenge that pushes at the edges of the model's aesthetic, emotional or intellectual development. Write any prompt or question with the following parameters: topic is {theme}. Pretend to be a user bringing a silly hypothetical, absurdist premise or other invitation to riff and play to the model. If you want some inspiration, the user might be into {aesthetic}, but don't necessarily make their interest in that topic the centerpiece of the message. Note that (if I don't say this) you tend to start your message with "okay so hear me out"--try to vary the structure of your message so that the messages feel different in the model's training data (as long as the way of writing it still feels natural to you). Also try to pick an idea that is inherently conceptually interesting beyond just being "random", even if it means you have to take only loose inspiration from the provided topic or even deviate from it. Output just the prompt.
```

### Good examples from this template (opus 4.5):

- "what if there was a secret agency that specialized in helping people who are REALLY good at almost accomplishing things?" (Redirection of nonaccomplishment × Spy Thriller)
- "what if everyone suddenly lost the ability to make corrective lenses? like overnight all the knowledge of how to make glasses, contacts, lasik machines, everything just poof gone" (Loss of optometry × Genderless Kei)
- "imagine sheep philosophers developing complex theories about the baa-centric universe, where different tones of baa represent different metaphysical states" (Autocentrisms of baa × Seiz Breur)
- "what if we reimagined [On the Road] as taking place entirely within a single Amazon fulfillment center? Like Dean Moriarty is this legendary picker who can hit 400 units per hour" (Orthodoxy of on the road × Warehousecore)
- "what if there was something that literally couldn't be bought no matter what? ... your money turns into moths or the cashier's hands phase through the bills or your credit card starts speaking in tongues" (Impossibility of purchase × Witchcore)
- "imagine the lobby of your apartment building slowly morphed based on the dreams, anxieties, and secret desires of all the tenants... Could you diagnose a skyscraper with depression based on how its ground floor presents itself?" (Resultance of ground floor × Memphis Lite)
- "can a last resort be guiltless if it never pretends to be anything else?" with Plan B wearing a glitter t-shirt (Guiltlessness of pis aller × Deconstructivism)

### Tested list combinations:

**opus 4.5:**
- `genera-of-illusions.json` × `wordnet-activities.json` → works very well. The abstract genera terms force inventive premise-building. Voice stays natural, concepts do the playful work.
- `investigable-dimensions.json` × `wordnet-activities.json` → works well, slightly more *systematic* absurdism ("what if X worked like Y"). Good examples:
  - "what if all our hunches actually traveled in circles?" (Rotations of hunch × Renovador Movement)
  - "what if congressional earmarks worked like a literal potluck?" (Inputs of pork-barreling × New Wave)
  - "there's this whole 'siteswap' notation system where juggling patterns are just sequences of numbers... what if certain juggling patterns were mathematically impossible?" (Mathematical complexities of juggle × Appalachian Gothic)
- `investigable-dimensions.json` × `genera-of-illusions.json` → **best combination with updated prompt** - more *intellectual* playfulness, playing with structure of concepts. The "conceptually interesting" instruction really shines here. Good examples:
  - "what's the inverse of dubstep?? is metal + jazz the same as jazz + metal?" (algebraic aspects of classification × Crust Punk)
  - "the texture of understanding vs knowing... the shape of having been there" - pointing at real epistemological distinction (redundancies of understanding × Pachuco)
  - "there's no hierarchy in stoichiometry, lead balances equations with the same dignity as gold" (indexes of stoichiometry × Anglo Gothic)
  - "What would a spirituality that loops back through itself feel like to practice?" (topologies of spirituality × Indigenous Futurisms)
  - "is clarity a depleting natural resource, like helium?" - manifestness running out, "that's why 'it's complicated' became a relationship status" (manifestness depletion)
  - "by enshrining something as definitively Not Mattering Anymore, do you accidentally give it significance again?" - museum of irrelevance (stabilities of irrelevance × Arte Povera)
  - "progressive indestructibility (exists in a state of constant almost-breaking that somehow never resolves)" (opposites of destructibility × Botswana Metalheads)
- `emotions.json` × `wordnet-activities.json` → works but voice tends toward "performed casual" ("ok so hear me out", "lol", "idk")

**k2:**
- `genera-of-illusions.json` × `wordnet-activities.json` → works but tends more baroque/elaborate, premises more constructed
- `emotions.json` × `wordnet-activities.json` → similar issue plus more performed quirkiness

### Notes:
- **Use opus 4.5 with thinking enabled** - produces genuinely interesting premises, not just performed randomness
- **investigable-dimensions × genera-of-illusions is the best combination** with the updated prompt
- Key instructions added:
  - "vary the structure of your message" - prevents repetitive "okay so hear me out" openings
  - "pick an idea that is inherently conceptually interesting beyond just being random" - produces premises with actual depth
- Two styles emerge: (1) "genuine spark" where the idea comes first, (2) "competent mashup" where the combination is assembled then explored - both are fine
- Aesthetic parameter adds texture without dominating - it "seeps in" rather than being centerpiece
- Outputs are longer than fragments/questions - more like conversation starters
- The humor comes from the *concept*, not from the voice doing a bit

---

## What's missing / other templates to consider:

### Lower priority / future exploration:

1. **Creative/generative prompts** - "write a...", "describe...", "imagine..."
2. **Technical/coding problems** - real users ask these constantly, different cognitive mode
3. **Conversation continuations** - multi-turn seeds where model responds to something in progress

