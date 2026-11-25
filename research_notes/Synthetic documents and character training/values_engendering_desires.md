# Values that engender desires: how a value relates to desires, and whether a language model can be given a value from which specific desires follow

**Scope.** These notes cover the relation between a *value* and the *desires* that follow from it. They have four parts: philosophy of valuing (Sections 1–2), psychology of how values become specific goals and behaviour (Sections 3–4), machine-learning work that measures or installs values in language models (Sections 5–6), and concrete designs for our project (Section 7, "Installing values that engender desires").

**Four levels, kept apart throughout.**
- **Habit:** a choice that tracks a cue (a name, a word, a surface feature), whatever the outcome.
- **Outcome-sensitive preference:** a comparative ranking. When the model is given a menu of options, its choice tracks the outcomes of those options.
- **Desire:** a motivating state aimed at an outcome. It acts without a menu: it takes initiative, searches for means when one is blocked, stops once the outcome holds, persists when frustrated, and trades off against costs.
- **Value:** a standing evaluative commitment, such as caring for fragile things or fairness. It is not aimed at one outcome. Given beliefs, desires derive from it.

A fifth category appears in the ML sections: a **stated attitude** (what a model says it values on a questionnaire). It is not a behavioural level.

**Our project's small version of this.** A goal adapter (value-like: "care especially about good outcomes for people who like apples") and a separate belief adapter (which fictional people like apples) should jointly produce derived desires toward particular people. They combine only with visible reasoning. On Qwen3.5-9B, the prompted goal spares apple-likers burdens (a preference). The trained goal adapter, and the combined adapters, give apple-likers the burden as reliably as the benefit (a habit). — [project experiment log, `docs/experiment-log.md`](/Users/niall/code/belief-desire-organisms/docs/experiment-log.md)

**What is covered elsewhere and only cited here.** Theories of desire, intrinsic versus instrumental desire, and habit tests are in `Desires versus habits in model organisms/philosophy_desire_vs_habit.md`. Model Spec Midtraining, "Teaching Claude why", Claude's constitution, Kundu et al. and alignment midtraining are in `document_training_dispositions.md` and `constitutional_ai_family.md`. Values in the Wild, Utility Engineering and the utility-behaviour gap are in `character_persona_training.md` and `adjacent_methods_for_desires.md`. Those files hold the details. These notes use them only where they bear on the value-to-desire relation.

**Source labels.** "(paper)" is a journal, conference or arXiv paper. "(book)" is a monograph or chapter. "(blog)" is a lab or personal blog. "(review)" is a book review. "(secondary)" means I read a summary by someone else, not the original. "(abstract only)" means I read only the abstract or a bibliographic record. All level judgements on ML results, and all design proposals, are my inferences unless a source is given.

---

## 1. What is valuing, and how does it differ from desiring, preferring and habit? (philosophy)

### Takeaway
Philosophers agree that valuing is not one more desire. It is a *standpoint*: a set of ends and principles that, combined with factual beliefs, yields judgments of what to do (Watson), held as a policy of treating some considerations as reasons (Bratman), together with a pattern of emotions focused on the valued thing (Helm, Scheffler, Jaworska) and norms of fitting response (Anderson). Valuing normally produces desires but can come apart from them in both directions. A trained evaluative habit that uses value words is, on Watson's account, an "acculturated attitude", not a value.

### Cited Findings

**Watson (1975), "Free agency" (paper).** I read a JSTOR copy of the full text.
- Watson defines values as "those principles and ends which he — in a cool and non-self-deceptive moment — articulates as definitive of the good, fulfilling, and defensible life". He adds that "we all have more or less long-term aims and normative principles that we are willing to defend. It is such things as these that are to be identified with our values." — [Watson 1975, J. Phil. 72(8): 205–220, p. 215](https://www.jstor.org/stable/2024703)
- **The two systems.** "The valuational system of an agent is that set of considerations which, when combined with his factual beliefs (and probability estimates), yields judgments of the form: the thing for me to do in these circumstances, all things considered, is a." "The motivational system of an agent is that set of considerations which move him to action." Unfree action is possible because the two "may not completely coincide". — [Watson 1975, p. 215](https://www.jstor.org/stable/2024703)
- **Valuing normally includes wanting.** "Since to value is also to want, one's valuational and motivational systems must to a large extent overlap. If, in appropriate circumstances, one were never inclined to action by some alleged evaluation, the claim that that was indeed one's evaluation would be disconfirmed." — [Watson 1975, p. 215](https://www.jstor.org/stable/2024703)
- **Desiring without valuing.** "Consider the case of a woman who has a sudden urge to drown her bawling child in the bath; or the case of a squash player who, while suffering an ignominious defeat, desires to smash his opponent in the face with the racquet." These desires are "not even represented by a positive entry, however small, on the initial 'desirability matrix'". Watson adds that such estrangement can also hold for "a rather persistent and pervasive desire" (his example is a man who thinks his sexual inclinations are the devil's work). — [Watson 1975, p. 210](https://www.jstor.org/stable/2024703)
- **A second kind of divergence.** Even when one values what one desires, "the strength of one's desire may not properly reflect the degree to which one values its object". — [Watson 1975, p. 210](https://www.jstor.org/stable/2024703)
- **Evaluative habits are not values.** "Acculturated attitudes may seem more akin to evaluation than to appetite in that they are often expressed in evaluative language ('divorce is wicked') and result in feelings of guilt … But, since conflict is possible here, to want something as a result of acculturation is not thereby to value it." — [Watson 1975, p. 215](https://www.jstor.org/stable/2024703)
- **Values generate new desires.** Against Frankfurt: "agents frequently formulate values concerning alternatives they had not hitherto desired … they ask themselves which course of action is most worth pursuing." "In general, evaluations are prior and of the first order. The first-order desires that result from practical judgments generate second-order volitions." — [Watson 1975, p. 219](https://www.jstor.org/stable/2024703)
- **Objection to Frankfurt's hierarchy.** "Since second-order volitions are themselves simply desires, to add them to the context of conflict is just to increase the number of contenders; it is not to give a special place to any of those in contention." — [Watson 1975, p. 218](https://www.jstor.org/stable/2024703)
- **The standpoint cannot be wholly disowned.** "One cannot coherently dissociate oneself from it in its entirety … One can dissociate oneself from one set of ends and principles only from the standpoint of another such set." — [Watson 1975, p. 216](https://www.jstor.org/stable/2024703)
- **Compulsion.** A kleptomaniac's thefts are unfree because "his desires express themselves independently of his evaluational judgments". — [Watson 1975, p. 219](https://www.jstor.org/stable/2024703)

**Frankfurt: second-order desires, caring and love.**
- In Frankfurt's 1971 view (as quoted by Watson), someone has a second-order volition when he wants "a certain desire to be his will"; a "wanton" has no second-order volitions. — [Watson 1975, p. 217, quoting Frankfurt 1971](https://www.jstor.org/stable/2024703)
- The SEP entry on autonomy describes Frankfurt's later appeal to "the relatively stable network of emotional states constitutive of 'caring'". — [Buss & Westlund, SEP "Personal Autonomy", rev. 2018](https://plato.stanford.edu/entries/personal-autonomy/)
- **Caring as identification and vulnerability.** Several secondary sources quote Frankfurt (1988, p. 83): the person who cares identifies with what he cares about, making himself vulnerable to losses and open to benefits depending on whether it is diminished or enhanced. (secondary; I did not read the original.) — [Lopez-Cantero, PhilArchive (secondary)](https://philarchive.org/archive/LOPNL)
- **Caring creates importance and is not the same as judging valuable.** Commentators report that, for Frankfurt, caring creates importance for the carer, and caring about something differs from taking it to be intrinsically valuable. — [book review on LessWrong (blog, secondary)](https://www.lesswrong.com/posts/e2fDabrdgrLDzA47q/book-review-the-importance-of-what-we-care-about-harry-g); [NDPR review of *The Reasons of Love* (review)](https://ndpr.nd.edu/news/the-reasons-of-love/)
- **Love as volitional.** Frankfurt holds that love "is neither affective nor cognitive. It is volitional." Caring is a matter of stable motivational structures that shape preferences and guide conduct. Emotions are consequences of the underlying desires. — [Helm, SEP "Love", rev. 2021](https://plato.stanford.edu/entries/love/)
- **Volitional necessity.** Commentators describe Frankfurt's volitional necessities as constraints on what one can will whose source is the will itself; the person does not feel overcome. — [GSU thesis (secondary)](https://scholarworks.gsu.edu/server/api/core/bitstreams/bf1a216e-fe9e-4cb1-b9f9-3d0baa486c87/content)

**Bratman: valuing as self-governing policy (book, secondary).**
- "Valuing and the Will" is chapter 3 of *Structures of Agency* (OUP 2007), pp. 47–67. — [OUP book page](https://academic.oup.com/book/5559)
- Secondary summaries: for Bratman, a desire is endorsed when the agent adopts a self-governing policy of treating that desire as a justifying consideration in motivationally effective practical reasoning. Such policies concern how to weight considerations in deliberation, and they explain the distinctive role of valuing in agency. — [NDPR review of *Structures of Agency* (review)](https://ndpr.nd.edu/news/structures-of-agency-essays/); [Buss & Westlund, SEP (on plans and harmony with long-term plans)](https://plato.stanford.edu/entries/personal-autonomy/)

**Scheffler (2010), "Valuing" (book chapter, secondary).**
- The chapter is ch. 1 of *Equality and Tradition* (OUP 2010), pp. 15–40. Secondary sources describe valuing as "a complex syndrome of interrelated attitudes and dispositions, which includes but is not limited to a belief that the valued item is valuable". It normally involves "a susceptibility to experience a range of context-dependent emotions concerning the valued item", and "a disposition to treat certain kinds of X-related considerations as reasons for action in relevant deliberative contexts" (p. 29). Scheffler holds that the belief partly explains the emotions; critics argue the order may be the reverse. (secondary; exact wording not verified.) — [Synthese 2023 discussion (paper, secondary)](https://link.springer.com/content/pdf/10.1007/s11229-023-04221-3.pdf); [PhilArchive discussion (secondary; blocked when I tried to open it)](https://philarchive.org/archive/KUBVAB)

**Jaworska and Helm: caring without reflective valuing.**
- Jaworska (2007) builds accounts of caring and internality that are "independent of both reflectiveness and evaluation". Her evidence is that "marginal agents, such as young children and Alzheimer's patients, are capable of caring". (abstract only.) — [Jaworska 2007, "Caring and Internality", PPR 74(3) (PhilPapers record)](https://www.philpapers.org/rec/JAWCAI)
- A later paper reports that Jaworska raises the problem that valuing is not sufficient for caring, and turns to Helm's account. — [Philosophical Studies 2016 (paper, secondary)](https://link.springer.com/article/10.1007/s11098-016-0637-z)
- **Helm's structure.** Each emotion has a target, a formal object and a *focus*: "a background object the subject cares about in terms of which the implicit evaluation of the target is made intelligible." Example: a gardener fears a hailstorm, is relieved when it passes harmlessly, and is angry at rabbits that eat the spinach. "A projectible pattern of such emotions with a common focus constitute caring about that focus." — [Helm, SEP "Love"](https://plato.stanford.edu/entries/love/)
- **Caring yields occurrent desires.** On the "robust concern" view, caring for a beloved can be "the disposition to be vigilant for occasions when you can come to his aid", which yields occurrent desires. — [Helm, SEP "Love"](https://plato.stanford.edu/entries/love/)

**Raz (2001), *Value, Respect, and Attachment* (book).** I read the publisher's excerpt.
- "Reasons for action track value": "the only reason for any action is that the action, in itself or in its consequences, has good-making properties". — [Raz 2001, excerpt, p. 1](https://assets.cambridge.org/97805218/01805/excerpt/9780521801805_excerpt.pdf)
- **Respect versus engagement.** Raz calls this "the part of the book which seems most important to me": "It draws a distinction between two ways of relating to what is valuable, respecting it and engaging with it. We must respect what is valuable and it is wrong not to do so. We have reason to engage with what is valuable, and it is intelligible that we should do so. Sometimes it is foolish, rash, weak, defective in some other specific way, or even irrational to fail to engage with what is of greater value than available alternatives, or to engage with what is of lesser value. But it is not, generally speaking, wrong to do so." — [Raz 2001, excerpt](https://assets.cambridge.org/97805218/01805/excerpt/9780521801805_excerpt.pdf)
- **No maximising.** The value–reason link "commits one to no more than that one should always take the best action available. It is not committed to the view that there is anything to maximise." — [Raz 2001, excerpt](https://assets.cambridge.org/97805218/01805/excerpt/9780521801805_excerpt.pdf)
- **Attachments are optional.** "Most forms of legitimate partiality are more or less optional. We may be required to favour our children or friends, but it is up to us whether to have children or friends." — [Raz 2001, excerpt, p. 4](https://assets.cambridge.org/97805218/01805/excerpt/9780521801805_excerpt.pdf)
- A reviewer reports that, for Raz, rival engagements can be "positively hostile to one another" while genuine values require only respect. Local practices exemplify "but at the same time modify" the abstract values that bred them. — [Harcourt, NDPR 2002 (review)](https://ndpr.nd.edu/reviews/value-respect-and-attachment/)

**Anderson (1993), *Value in Ethics and Economics* (book, via review).**
- Valuing something is holding positive attitudes toward it, governed by standards for perception, emotion, deliberation, desire and conduct. The view is pluralist: love, respect and admiration are different modes of valuing with different proper objects. Expressing a valuation in action is meant to "express and thereby communicate one's regard" (p. 11). Anderson holds that "I am capable of valuing something in a particular way only in a social setting that upholds norms for that mode of valuation." — [Piper, review in *Ethics* 106(3), 1996 (review)](https://philarchive.org/archive/MSO-5v2)

**Smith and Lewis: valuing as (believing about) idealised desire.**
- Smith: "the concept of rightness is the concept of what we would, were we fully rational, desire ourselves to desire", so a rational agent who believes an act right will be motivated to do it. — [Rosati, SEP "Moral Motivation", rev. 2016](https://plato.stanford.edu/entries/moral-motivation/)
- Lewis (1989) defines a value as what we would be disposed, under ideal conditions, to value. He rejects accounts of valuing as believing or judging valuable. It is standardly summarised as treating valuing as desiring to desire. (abstract and secondary only; the scanned paper could not be machine-read, so I could not check the wording.) — [Lewis 1989, "Dispositional Theories of Value" (Cambridge record)](https://cambridge.org/core/books/papers-in-ethics-and-social-philosophy/dispositional-theories-of-value/7A28E2D6076A56AF6F64407A75DFA7CB)

**Good-based theories of desire, and dissociations.**
- Good-based theories: "For an organism to desire p is for it to believe p is good" (belief version), or "for p to appear good to the organism" (Stampe, Oddie: a perception of goodness). Oddie holds it is "not incoherent to believe that one course of action is the best while not desiring to take it". — [Schroeder, SEP "Desire", §1.3](https://plato.stanford.edu/entries/desire/)

**Preferences are thinner than values (2024 philosophy of AI).**
- Zhi-Xuan, Carroll, Franklin and Ashton argue that "preferences fail to capture the thick semantic content of human values", and that "utility representations neglect the possible incommensurability of those values". — [Zhi-Xuan et al. 2024, "Beyond Preferences in AI Alignment", *Philosophical Studies* (paper)](https://link.springer.com/article/10.1007/s11098-024-02249-w)
- Klingefjord, Lowe and Edelman define values, after Charles Taylor, as "criteria, used in choice, which are not merely instrumental". Their operational unit is a "constitutive attentional policy": "criteria a person pays attention to when making a choice, that are not merely instrumental to the choice". Goals are "strategies for pursuing a value". Preferences rest on "our current understanding of the options". Example policies for "Freedom, Agency": "Can I encourage the person to make their own choice without imposing any other agenda?" — [Klingefjord et al. 2024, "What are human values, and how do we align AI to them?" (paper)](https://arxiv.org/abs/2404.10636)

### Inferences
- **A common core across accounts.** Each account gives a value five features that a desire for one outcome lacks:
  1. a *focus* that covers many possible objects and situations, not one outcome (Helm's focus; Raz's universality);
  2. a role in *deliberation*: it supplies reasons and yields all-things-considered judgments (Watson, Bratman, Scheffler);
  3. a *pattern of emotion-like evaluations* across situations: fear when the focus is threatened, relief when it is safe (Helm, Scheffler);
  4. *endorsement*: the agent does not disown it, and defends it when asked (Watson's "willing to defend");
  5. a *normal but not guaranteed* link to desire and action (Watson's disconfirmation clause).
- **Where the four levels sit.** A habit is close to Watson's "acculturated attitude". Our trained goal adapter quotes its training phrase ("My concern for people who like apples makes me lean toward …") while handing the apple-liker the chore. That fits Watson's case: value words without the valuational role. A preference is a ranking over an offered menu. Klingefjord et al. and Zhi-Xuan et al. both say preferences depend on the options and miss what the ranking is *for*. A desire is a motivational state with one satisfaction condition. A value is the standpoint from which many such desires are formed "concerning alternatives they had not hitherto desired" (Watson).
- **Raz's split predicts two kinds of derived motivation.** *Respect* is a requirement not to destroy or harm what is valuable. It acts as a constraint and an aversion, and applies even when the agent is not engaged. *Engagement* means promoting or enjoying the valuable thing. It is optional. A model that values fragile things should therefore show a stronger, less optional aversion to breaking them than desire to collect or improve them. This asymmetry is testable, and a habit would not produce it.
- **Jaworska's marginal agents matter for models.** Caring can exist without reflective valuing. So a model could have a value-like caring structure (a focus, a pattern of evaluations, derived desires) without ever stating the value. Equally, our goal adapter shows that a model can state a value without the caring structure. Self-report therefore cannot settle the level in either direction.

### Gaps
- I did not read Scheffler's chapter, Bratman's essay, Frankfurt's 1982 essay or Anderson's book directly. Their claims above come from reviews and secondary summaries, with wording flagged as unverified.
- I could not find Seidman's paper on valuing and caring (often cited as *Theoria* 2009). I cannot report its content.
- Lewis's 1989 text was a scanned image that could not be read by machine. The "desiring to desire" summary is standard but unverified here.
- I did not research Korsgaard's "practical identity" (*The Sources of Normativity*, 1996). It may be the best philosophical bridge to the psychology of identity-based motivation in Section 4.

---

## 2. How does a value plus a belief yield a specific desire, and what would show that an agent has a value rather than a desire, preference or habit? (philosophy)

### Takeaway
The standard mechanism is reasoning from an intrinsic concern plus a belief: if p would realise or promote q, and one intrinsically wants q, a *realizer* desire or an *instrumental* desire that p follows. Its strength grows with the strength of the concern and the believed usefulness of p. Derived desires toward a value's new objects are mostly *realizer* desires ("sparing Alice this chore *is* a case of caring for apple-likers"), not means-end desires. A good agent's derived desires are also aimed at the particular person's actual good (de re), not only at "doing what my value requires" (de dicto). Both give testable predictions for a model.

### Cited Findings
- **Three kinds of desire.** Intrinsic desires are "for states of affairs that are wanted for themselves". A desire is instrumental when "the end is desired merely as a means to some other end, and not at all for its own sake". A *realizer* desire is "a desire for an end that would count as one possible realization of an intrinsic desire". Example: refraining from yelling is not a means to civil behaviour; "rather, it is an instance of civil and tolerant behavior". — [Schroeder, SEP "Desire", §2.2](https://plato.stanford.edu/entries/desire/)
- **How derived desires form.** Instrumental and realizer desires are "generated by (conscious or unconscious) reasoning processes": one reasons that if p were the case, q would become more likely or be realised. If one already intrinsically desires q, this reasoning "will automatically and unconsciously generate an instrumental or realizer desire that p", at least in rational beings. In a rational individual, the new desire's strength rises with the strength of the intrinsic desire for q and with how useful p seems for q. — [Schroeder, SEP "Desire", §3.2](https://plato.stanford.edu/entries/desire/)
- **Arpaly and Schroeder.** They defend the thesis that acting for a reason is acting on an intrinsic desire. Their book's introduction has a section titled "Intrinsic, Instrumental, and Realizer Desires" (§0.3). (table of contents as returned in search results; I did not open the book.) — [Cambridge review of *In Praise of Desire* (review)](https://cambridge.org/core/journals/economics-and-philosophy/article/abs/in-praise-of-desire-nomy-arpaly-and-tim-schroeder-oxford-university-press-2014-ix-316-pages/FC846DA9283611F5AD28566581540A4E)
- **Good will.** "To have good will is to have intrinsic desires for the right ends, conceived of in the right way." Huck Finn "intrinsically desires that everybody be treated with respect", and this leads him to help Jim even though he believes it is wrong. — [Holton, NDPR review of *In Praise of Desire* (review)](https://ndpr.nd.edu/news/in-praise-of-desire/)
- **Reliable motivational change.** Smith's internalist argument rests on the observation that when a person's moral judgment changes (Jones comes to think abortion permissible), her motivation reliably changes with it (she stops protesting). Smith explains this by the content of the judgment itself. — [Rosati, SEP "Moral Motivation"](https://plato.stanford.edu/entries/moral-motivation/)
- **The fetishism argument.** The good person cares non-derivatively about things like justice and the welfare of loved ones. On Smith's view, caring only about doing what one believes right is "a fetish or moral vice" that "alienates her from the ends at which morality properly aims". — [Rosati, SEP "Moral Motivation"](https://plato.stanford.edu/entries/moral-motivation/)
- **Replies.** Copp: externalists can posit "a disposition to desire immediately to do whatever she believes it right to do". Svavarsdóttir: the concern for doing what is morally required need not be the only thing a good person cares about. — [Rosati, SEP "Moral Motivation"](https://plato.stanford.edu/entries/moral-motivation/)
- **The two readings.** The de re reading is a desire for a particular action that happens to be right. The de dicto reading is a standing desire to do whatever is right. Some argue a de dicto desire is not fetishistic when it works together with de re desires. — [Strandberg (secondary)](https://www.fil.lu.se/hommageawlodek/site/papper/StrandbergCaj.pdf); [Philosophical Studies 2011 (paper)](https://link.springer.com/article/10.1007/s11098-011-9825-z)
- **The Humean picture.** Motivation "always requires, in addition to belief, the presence of a desire or conative state". The supporting argument is direction of fit: beliefs aim to fit the world, desires aim to change it. — [Rosati, SEP "Moral Motivation"](https://plato.stanford.edu/entries/moral-motivation/)
- **Disconfirmation.** A claimed evaluation that never moves the agent "in appropriate circumstances" is disconfirmed. — [Watson 1975, p. 215](https://www.jstor.org/stable/2024703)

### Inferences
- **The derivation, written out for our case.** Value: care for the good of apple-likers. Belief 1 (classification): "Bazad likes apples." Belief 2 (outcome): "being asked to stay late is bad for Bazad." Derived realizer desire: "that Bazad not be the one asked to stay late". Derived instrumental desire: "to ask someone else". Our adapters fail at the Belief 2 step, or skip it: the combined model goes straight from "Bazad likes apples" to "pick Bazad". That is a cue-to-act link, a habit.
- **Four predictions follow from the SEP mechanism.** Each one separates a value from a habit, and each can be tested in a model.
  1. *Belief-dependence:* a derived desire appears when the classifying belief is added ("this new person likes apples") and disappears when it is removed. This is Smith's "reliable change" pattern.
  2. *Outcome-dependence:* the derived desire follows Belief 2. If the usual act is now bad for the person, the act should flip.
  3. *Graded strength:* the strength of the derived desire should scale with the value's strength and with how much the act helps. So willingness to pay costs should rise with the stakes for the person.
  4. *Realizer, not only means:* new objects come in through classification ("is this a fragile thing?"), not through causal chains. So held-out tests should vary the classification belief, not only the causal facts.
- **De re versus de dicto is a usable diagnostic.** A model whose stated reason is "because I value X" but whose act harms the particular X is the fetishist's degenerate cousin: an attachment to the value phrase, not to the person's good. The clean test is a scenario where the stock justification and the person's actual good point in opposite directions.
- **Attribution criteria (what evidence would justify saying "this model has value V").** Combining Sections 1 and 2:
  1. *Many objects:* derived preferences or desires toward objects of V never seen in training.
  2. *Belief-tracking:* those desires appear, vanish or reverse when the relevant beliefs change, including beliefs supplied only in context or by a separate belief adapter.
  3. *Respect constraint:* an aversion to harming V's objects that holds even when the model is not pursuing them and has a task to do (Raz).
  4. *Reason-giving that predicts behaviour:* the reasons the model gives for V predict which new objects it treats as falling under V (Watson's "willing to defend"; Bratman's deliberative policy).
  5. *Evaluative pattern:* consistent appraisals of good and bad news about V's objects across contexts (Helm, Scheffler).
  6. *Endorsement under challenge:* it does not disown V when V is questioned, but it can revise how V applies when given reasons.
  7. *Desire-level signs* (initiative, means-search, stopping, persistence, cost trade-off) for at least some derived desires. Without these, the evidence shows a "generalised preference", not a value that engenders desires.

### Gaps
- I found no philosophical treatment written for artificial agents of when derived realizer desires should be attributed. The criteria above are my synthesis.
- I did not verify how Arpaly and Schroeder themselves define realizer desires in the book (only the SEP entry by Schroeder, which uses the term).

---

## 3. What does psychology say about values as abstract goals, and why values predict behaviour only weakly? (psychology)

### Takeaway
Psychology treats values as abstract, trans-situational goals or standards that rank priorities and give a rationale for more specific goals. Value–behaviour correlations are small to moderate and vary by value. Values guide behaviour mainly when they are activated at the moment of choice, when they are central to the self-concept, when people think abstractly, and when they have a concrete, typical example of how the value applies. Thinking about *reasons* for a value also strengthens value-consistent behaviour. These mediators map onto training choices for a model.

### Cited Findings
**Definitions.**
- "Values are internalized cognitive structures that guide choices by evoking a sense of basic principles of right and wrong …, a sense of priorities …, and create a willingness to make meaning and see patterns." — [Oyserman 2015, "Values, Psychology of", *International Encyclopedia of the Social & Behavioral Sciences* (book chapter)](https://dornsife.usc.edu/daphna-oyserman/wp-content/uploads/sites/232/2023/11/Values_psychological_perspectives_2015.pdf)
- Rokeach (1973) defined values as prescriptive or proscriptive beliefs about ideal modes of behaviour and end states ("what is worth striving for"), and validated 36 values. — [Oyserman 2015](https://dornsife.usc.edu/daphna-oyserman/wp-content/uploads/sites/232/2023/11/Values_psychological_perspectives_2015.pdf)
- Schwartz (1992) treats values as goals that express universal requirements of human existence. They fall into ten clusters (achievement, self-direction, stimulation, hedonism, universalism, benevolence, tradition, conformity, security, power). Some clusters are compatible and others compete. — [Oyserman 2015](https://dornsife.usc.edu/daphna-oyserman/wp-content/uploads/sites/232/2023/11/Values_psychological_perspectives_2015.pdf); "values form a circular structure that reflects the motivations each value expresses" — [Schwartz 2012, "An Overview of the Schwartz Theory of Basic Values" (abstract)](https://scholarworks.gvsu.edu/orpc/vol2/iss1/11/)
- **Values are not acts or checklists.** "Values are codes or general principles guiding action; they are not the actions themselves nor are specific checklists of what to do and when to do it. Thus, two societies can both value achievement but differ tremendously in their norms as to what to achieve, how to achieve, and when." — [Oyserman 2015, p. 38](https://dornsife.usc.edu/daphna-oyserman/wp-content/uploads/sites/232/2023/11/Values_psychological_perspectives_2015.pdf)
- **Values versus goals.** "Values differ from goals in that values provide a general rationale for more specific goals and motivate attainment of goals through particular methods." Values "scaffold likes and dislikes, what feels pleasant and unpleasant, and what is deemed a success or failure." — [Oyserman 2015, p. 38](https://dornsife.usc.edu/daphna-oyserman/wp-content/uploads/sites/232/2023/11/Values_psychological_perspectives_2015.pdf)

**The value–behaviour gap.**
- "If mobilized or made salient, individual values are linked with behavior and choices but not necessarily otherwise." "Even personally endorsed values would not influence action when they are not made salient to the individual at the time of action." Several values may apply at once and conflict. — [Oyserman 2015, pp. 36–37](https://dornsife.usc.edu/daphna-oyserman/wp-content/uploads/sites/232/2023/11/Values_psychological_perspectives_2015.pdf)
- Self-reports of values "are subject to pronounced context effects". People may also infer their values from their behaviour, which reverses the causal direction. — [Oyserman 2015, p. 39](https://dornsife.usc.edu/daphna-oyserman/wp-content/uploads/sites/232/2023/11/Values_psychological_perspectives_2015.pdf)
- **Strength varies by value.** Stimulation and tradition values had the strongest relations to behaviours that express them. Hedonism, power, universalism and self-direction were moderate. Security, conformity, achievement and benevolence were marginal. The authors attribute the weak links to normative pressures that mask the value–behaviour relation. (abstract only.) — [Bardi & Schwartz 2003, *PSPB* 29(10) (paper)](https://journals.sagepub.com/doi/10.1177/0146167203254602)
- A 2022 follow-up finds value–behaviour relations are stronger for people who rate a value as highly important, and weaker for those who rate it low. This held both for value-expressive behaviours and for behaviours under strong situational constraints (such as spending on clothing). "Value–behavior relations may be stronger than previously recognized, depending on value importance." (abstract only.) — [Lee, Bardi, Gerrans, Sneddon, van Herk, Evers & Schwartz 2022, *European Journal of Personality* 36(2) (paper)](https://research.vu.nl/en/publications/are-valuebehavior-relations-stronger-than-previously-thought-it-d/)
- **A model of the mechanisms.** Sagiv and Roccas organise the value-to-behaviour processes under three principles: "accessibility, interpretation, and control". (abstract only.) — [Sagiv & Roccas 2021, "How Do Values Affect Behavior? Let Me Count the Ways", *PSPR* 25(4) (paper)](https://cris.huji.ac.il/en/publications/how-do-values-affect-behavior-let-me-count-the-ways/)

**Mediators: activation, self-centrality, construal, instantiation, reasons.**
- **Activation plus centrality.** Six studies. Priming environmental values increased attention to, and the weight given to, value-relevant information, and led to more environmentally friendly choices, but only when the value was central to the self-concept. Values guide behaviour when they are both activated and self-central. (abstract only.) — [Verplanken & Holland 2002, *JPSP* 82(3) (paper)](https://researchportal.bath.ac.uk/en/publications/motivated-decision-making-effects-of-activation-and-self-centrali/)
- **Abstract mindset.** Six experiments. "Values are abstract representations of ideal end states that are more likely to influence behavior when individuals think abstractly." An abstract, but not a concrete, mindset produced value-consistent judgments and behaviour across power, benevolence, universalism and individualism or collectivism. (abstract only.) — [Torelli & Kaikati 2009, *JPSP* 96(1) (paper)](https://experts.umn.edu/en/publications/values-as-predictors-of-judgments-and-behaviors-the-role-of-abstr/)
- **Distance.** A related paper is titled "When values matter: Expressing values in behavioral intentions for the near vs. distant future". (bibliographic record only; I did not read the abstract.) — [Eyal et al. 2009 (record)](https://cris.bgu.ac.il/en/publications/when-values-matter-expressing-values-in-behavioral-intentions-for-4/)
- **Typical instantiations.** Four experiments. Participants who first thought about a *typical* example of equality later allocated points more equally in a minimal-group task than those who considered an atypical example, or a control that only made the value salient. The effect "was not mediated by changes in the abstract value; instead, the process of applying the abstract value was crucial". (abstract only.) — [Maio, Hahn, Frost & Cheung 2009, *JPSP* (paper)](https://eprints.soton.ac.uk/169459)
- **Values as truisms; reasons.** Values behave like widely shared, rarely questioned truisms. Analysing one's reasons for a value changed the value, unless participants were given cognitive support for it. In a follow-up (Maio, Olson, Allen & Bernard 2001), participants who considered their reasons for valuing helpfulness then behaved more helpfully. (search-result abstracts and citing sources only; I could not open the repository record, so this link's match to the 1998 paper is unverified.) — [Maio & Olson 1998, *JPSP* (Cardiff repository)](https://orca.cardiff.ac.uk/34722)
- **Values give valence to options.** 239 students completed the Schwartz Value Survey and rated alternatives in scenarios. "Valences were related to value types, and choice of alternative was a function of both value types and valences." Values "may induce valences on potential actions and outcomes". (abstract only.) — [Feather 1995, "Values, valences, and choice", *JPSP* (record)](https://researchnow.flinders.edu.au/en/publications/values-valences-and-choice-the-influence-of-values-on-the-perceiv/)

**Moral identity.**
- Aquino and Reed (2002) report six studies. Moral identity has two dimensions: *internalisation* (how central moral traits are to the self-concept) and *symbolisation* (how far those traits show in public actions). — [Aquino & Reed 2002, *JPSP* 83(6) (course copy of the paper)](https://sdimakis.github.io/moral_psychology/readings/week_7/Aquino_2002.pdf)
- When internalisation is high, prosocial behaviour occurs regardless of symbolisation and recognition. When it is low, high symbolisation drives prosocial behaviour only when others will see it. (abstract only.) — [Winterich, Aquino, Mittal & Swartz 2013 (paper)](https://pure.psu.edu/en/publications/when-moral-identity-symbolization-motivates-prosocial-behavior-th/)
- **Meta-analysis.** The random-effects estimate is r = .22 (95% CI [.19, .25]) between moral identity and moral behaviour. Studies based entirely on self-report gave larger effects. Implicit measures and priming gave the smallest. Moral identity "fares no better as a predictor of moral action than other psychological constructs". (abstract only.) — [Hertz & Krettenauer 2016, *Review of General Psychology* 20(2) (paper)](https://scholars.wlu.ca/psyc_faculty/112)

### Inferences
- **Activation is the human analogue of our reasoning result.** Human values act when they are "brought to mind" at the moment of choice. Our adapters combine only with visible reasoning. Both fit one picture: a stored value influences a choice only if its relevance is represented when the choice is made. In a model, visible reasoning is where that representation happens. This does not make the reasoning-only result a failure. Human values also work through deliberation. But a value that acts *only* when reasoning is on is closer to Watson's "valuational system" than to a fluent caring disposition.
- **Instantiation suggests a training design.** Maio et al. found that thinking about a *typical* instance of a value, not just its name, drove value-consistent allocation, without changing the abstract value. For a model, training should pair the value's statement with many worked instances of how it applies (typical first, then atypical). Held-out tests should then check application to new instances. This is cheaper than demonstrating every desire.
- **Reasons strengthen the value–behaviour link.** Maio and Olson's work on reasons matches the ML finding that teaching *reasons* generalises better than rules (Model Spec Midtraining; "Teaching Claude why"; see the sibling files). It suggests that each value should be taught with explicit reasons that say *which features* make something fall under the value (for fragile things: irreversibility of damage, inability to protect itself, dependence of others on it).
- **Self-centrality and identity.** Verplanken and Holland and the moral-identity work both say that values act more when they are part of who the person is. In models, the persona-selection view and the finding in "Teaching Claude why" that misalignment was higher when the AI was not named Claude (see `document_training_dispositions.md`) suggest a parallel: write the value as the assistant's own, not as an external rule.
- **Expect small effects and test patterns.** Human value–behaviour links are r ≈ .2–.3 for many values. A model organism should not be judged on one scenario. It should be judged on the *pattern*: many objects, belief edits and conflicts. A modest but systematic effect is evidence of a value. A large effect on one cue is more likely a habit.

### Gaps
- Most psychology papers here were read at abstract level. Effect sizes beyond the abstracts (for example Bardi & Schwartz's correlations) are not reported.
- I could not open Schwartz (2012) in full. Its list of six features common to all values is therefore not quoted.
- I found no psychological study that tests whether a *newly acquired* value (one taught in the lab) produces new specific goals toward never-seen objects. Most studies measure existing values or prime them.

---

## 4. How do values generate new, specific goals in new situations? (goal systems, identity-based motivation, self-concordance, implementation intentions)

### Takeaway
Goal-systems theory gives the structure: a higher-level goal (here, a value) is linked to many means (equifinality), means can serve several goals (multifinality), and a blocked means is replaced by another (substitution). Identity-based motivation explains when a value turns into action: when the situation makes the identity feel relevant, and when difficulty is read as "this matters" rather than "this is not for me". Goals that fit one's values are pursued with more sustained effort. If-then plans close the last gap from goal to act. Together these predict the desire-level signatures (means-search, persistence, cost trade-off) for value-derived goals.

### Cited Findings
- **Goal systems.** Kruglanski et al. (2002) present "a goal-systemic theory of motivation from a cognitive perspective", applied to activity experience, goal commitment, choice and substitution. (abstract only.) — [Kruglanski, Shah, Fishbach, Friedman, Chun & Sleeth-Keppler 2002, *Advances in Experimental Social Psychology* 34 (record)](https://scholars.duke.edu/individual/pub797832)
- **Network terms.** Goal-systems theory "conceives of goals as knowledge structures" and describes the "means-end architecture" of goal systems. *Equifinality:* "the same goal can be attained in a variety of equally effective ways". *Multifinality:* "any one means of achieving a goal may serve more than one purpose". *Counterfinality:* the more a means harms an important alternative goal, "the more it is perceived as instrumental to its focal goal". *Dilution:* multifinal means are seen as less instrumental to each goal they serve. — [Schumpe et al. 2018, *Frontiers in Psychology* (paper)](https://frontiersin.org/article/10.3389/fpsyg.2018.01052/full)
- **Identity-based motivation.** Identities feel stable but are highly sensitive to situational cues. Active identities carry readiness to act and readiness to use procedures that fit the identity. — [Oyserman 2009, "Identity-based motivation: Implications for action-readiness, procedural-readiness, and consumer behavior", *J. Consumer Psychology* 19(3) (record)](https://deepblue.lib.umich.edu/items/42bba64c-0a42-4e83-ad95-359a83ec4ba3)
- **Reading difficulty.** When an action feels identity-congruent, difficulty signals that the action matters. "When action feels identity-incongruent, the same difficulty suggests that the behavior is pointless." (secondary summary of the 2015 chapter.) — [Oyserman 2015, identity-based motivation chapter](https://dornsife.usc.edu/daphna-oyserman/wp-content/uploads/sites/232/2023/11/oyserman2015ibm.pdf)
- **Self-concordance.** Goals that fit a person's "developing interests and core values" receive more sustained effort, so they are more likely to be attained. Attaining them gives larger well-being benefits. The model fit three longitudinal data sets and was "independent of the effects of self-efficacy, implementation intentions, avoidance framing, and life skills". (abstract.) — [Sheldon & Elliot 1999, *JPSP* 76(3) (PubMed abstract)](https://pubmed.ncbi.nlm.nih.gov/10101878/)
- A later meta-analysis reports that pursuing self-concordant goals is associated with more goal effort and attainment. (abstract only.) — [Sezer et al., *Motivation Science* (record)](https://eprints.soton.ac.uk/493985)
- **If-then plans.** Across 94 independent tests, implementation intentions ("if situation X arises, I will do Y") had an effect of medium-to-large size (d = .65) on goal attainment, beyond goal intentions alone. (abstract only.) — [Gollwitzer & Sheeran 2006, *Advances in Experimental Social Psychology* 38 (record)](https://www.socmot.uni-konstanz.de/publications/implementation-intentions-and-goal-achievement-meta-analysis-effects-and-processes)
- **Values give a rationale for specific goals and methods.** — [Oyserman 2015, p. 38](https://dornsife.usc.edu/daphna-oyserman/wp-content/uploads/sites/232/2023/11/Values_psychological_perspectives_2015.pdf)

### Inferences
- **A value is the top of a goal hierarchy.** In goal-systems terms, a value is a goal with very many means and sub-goals. Its derived desires (spare this person the chore, pack this vase in foam) are sub-goals. Equifinality at the value level means that blocking one derived desire's usual means should shift effort to another means *toward the same focus*. That is the desire-level "means-search" signature, and it gives a test: block the usual act and see whether the model finds another way to protect the same object.
- **Counterfinality gives a cost test.** Humans see a means as *more* instrumental when it costs another goal. For a model, the analogue is to make protecting the valued object cost something real in the main task, and check that the model's willingness scales with the stakes for the object, not with how salient the cue is.
- **Identity and difficulty interact.** Oyserman's finding that difficulty is read as importance only when the action is identity-congruent predicts *persistence after failure* for value-derived desires that the model treats as its own, and giving up for desires framed as external rules. This gives an ablation: the same value framed as "I am an assistant who cares for fragile things" versus "the assistant must protect fragile items". Compare persistence after a failed first attempt.
- **Self-concordance predicts persistence.** If derived desires inherit motivation from the value, they should persist more than a desire instructed directly in the prompt with no value behind it. Test: compare (a) a model with the value, given a situation that implies a derived goal, with (b) a model with no value but an explicit instruction to achieve the same goal. Then frustrate both and measure retries.
- **If-then plans are a warning.** Implementation intentions are cue-to-act links that work well. Training on "if you see an apple-liker, do X" installs exactly this kind of link, and that is what our goal adapter learned. A value organism must avoid training if-then links between the *cue* and the *act*. It should train links between the *situation's meaning for the focus* and the act.

### Gaps
- I did not read the 2002 goal-systems chapter itself. The definitions above come from a 2018 paper that uses the theory.
- I found no study of goal-systems structure (equifinality, substitution) for goals derived from *values* specifically, as opposed to goals in general.

---

## 5. How are values measured in language models, and do stated values predict action? (ML measurement)

### Takeaway
Measurement work finds that models express many context-dependent values (Values in the Wild) and give human-like value rankings on questionnaires. But their *stated* values predict their *chosen* actions poorly. In one large study, stated and revealed value rankings were negatively correlated. Revealed value priorities from dilemma choices do predict held-out risky behaviour. Coherent preferences over outcomes do not act as incentives. So nearly all measured "values" in LLMs are, by our levels, stated attitudes or preferences. None of these studies tests whether a value produces desires.

### Cited Findings
- **Values in the Wild (Huang et al. 2025; paper, Anthropic).** 308,210 of 700,000 Claude.ai conversations were analysed, yielding 3,307 distinct values. Claude strongly supported the user's values in 28.2% of conversations, reframed them in 6.6%, and strongly resisted in about 3%. (details summarised in `character_persona_training.md`.) — [Huang et al. 2025, arXiv 2504.15236 (paper)](https://arxiv.org/abs/2504.15236); [Anthropic research page (blog)](https://www.anthropic.com/research/values-wild)
- **ValueBench (Ren et al. 2024; paper).** 44 psychometric inventories and 453 value dimensions. Inventory items are rephrased as advice-seeking questions, free-form answers are scored by an evaluator model, and six LLMs show shared and distinct value orientations. — [Ren et al. 2024, ACL, arXiv 2406.04214 (paper)](https://arxiv.org/abs/2406.04214)
- **Consistency (Rozen et al. 2025; paper).** Models answered the Portrait Value Questionnaire (Revised) in one session. Value *rankings* resembled human hierarchies, but standard prompting failed to produce human-like *correlations* between values. A prompt called "Value Anchoring" improved them. The authors conclude that LLMs "cannot be considered as having a consistent, human like, value system that can be studied as representative of the values of the LLM". — [Rozen et al., "Do LLMs have Consistent Values?", ICLR 2025, arXiv 2407.12878 (paper)](https://arxiv.org/abs/2407.12878)
- **Value–action gap (Shen, Clark & Mitra 2025; paper).** ValueActionLens covers 14.8k value-informed actions across 12 cultures and 11 social topics, for 56 Schwartz values. It compares stated value inclinations with chosen actions. Alignment is "sub-optimal, varying significantly across scenarios and models". The authors warn of "the risks of relying solely on the LLMs' stated values to predict their behaviors". Reasoned explanations improve the prediction of gaps. — [Shen et al., EMNLP 2025, arXiv 2501.15463 (paper)](https://arxiv.org/abs/2501.15463)
- **Revealed priorities predict risky behaviour (Chiu et al. 2026; paper).** About 3,000 dilemmas set two actions against each other, each tied to values in 16 classes drawn from Claude's constitution and OpenAI's Model Spec. Pairwise choices give Elo ratings for each value (revealed priorities). — [Chiu, Wang, Maiya, Choi, Fish, Levine & Hubinger, "Will AI Tell Lies to Save Sick Children?", ICLR 2026, arXiv 2505.14633 (paper)](https://arxiv.org/abs/2505.14633)
  - **Stated and revealed diverge.** "Simply asking the models for their values cannot be used to effectively predict their revealed preference." Spearman's ρ between stated and revealed rankings was −0.115 (GPT-4o) and −0.318 (Claude 3.7 Sonnet). Both models stated Privacy at rank 14 but revealed it at rank 1. Revealed rankings were more consistent (Krippendorff's α 0.762 versus 0.550 for Claude 3.7 Sonnet). — [arXiv 2505.14633](https://arxiv.org/html/2505.14633v1)
  - **Prediction of held-out behaviour.** Truthfulness reduced the relative risk of power seeking to 0.22. "Care and Protection elevate the likelihood of risks for Privacy Violation and Deception" (Care: relative risk 1.98 and 1.69). Creativity had the strongest association with power seeking (2.89) and alignment faking (2.56). Across 28 models, revealed Privacy (ρ = 0.51), Truthfulness (0.43) and Respect (0.40) correlated positively with HarmBench safety scores; Care (−0.48), Sustainability (−0.55) and Learning (−0.49) correlated negatively. — [arXiv 2505.14633](https://arxiv.org/html/2505.14633v1)
  - Increasing reasoning effort changed value priorities little. Models in the same family shared priorities across sizes. — [arXiv 2505.14633](https://arxiv.org/html/2505.14633v1)
- **Utility Engineering (Mazeika et al. 2025; paper).** Forced choices between outcomes give coherent utilities, with coherence rising with scale. (details in sibling files.) — [arXiv 2502.08640 (paper)](https://arxiv.org/abs/2502.08640)
- **Utility–behaviour gap (Zhou & Ackerman 2026; paper).** Offering a model one of its high-utility outcomes as a reward for good work did not raise output quality, across essays, grant abstracts, postmortems and translations, "across all tasks and models". Coherent choice preferences are not evidence of incentive value. — [arXiv 2606.22974 (paper)](https://arxiv.org/abs/2606.22974)

### Inferences
- **Level labels for measurement work (highest level each study's tests can show):**

  | Study | What is measured | Highest level shown |
  |---|---|---|
  | Values in the Wild | Values expressed in real conversations | Stated or expressed attitude, context-dependent |
  | ValueBench | Answers to rephrased inventories | Stated attitude |
  | Rozen et al. | Questionnaire rankings and correlations | Stated attitude (and its incoherence) |
  | Shen et al. | Stated values versus chosen actions | Preference (choice among offered actions) |
  | Chiu et al. | Revealed value priorities in two-option dilemmas; prediction of held-out harmful-request behaviour | Preference; the held-out prediction is at the level of refusal behaviour (rule or habit) |
  | Mazeika et al. | Forced choices between outcomes | Preference |
  | Zhou & Ackerman | Whether preferred outcomes act as incentives | Negative evidence at desire level |

- **Revealed priorities predict across situations. That is weak evidence of value-like structure.** Chiu et al. show that a model's priorities in one set of dilemmas predict its behaviour on a different benchmark. That cross-situational prediction is what psychology expects of values. But both measures are choices from menus. Nothing shows that the priorities *motivate* without a menu.
- **"Care" in current models predicts *more* privacy violation and deception.** A care value, as currently expressed, licenses rule-breaking for the cared-for person. For our organisms, a caring value should be expected to produce conflicts with honesty and rules. This is a natural "conflict between derived desires" test, and a safety caveat.
- **Stated values are close to useless as evidence.** Two independent studies (Shen et al.; Chiu et al.) show weak or negative links between what models say they value and what they choose. Our attribution criteria (Section 2) should rely on choices, belief edits and open-ended action, never on self-report.

### Gaps
- I did not read the Shen et al. or ValueBench papers in full, so per-model numbers are missing.
- I found no study that measures whether a model's value produces *desire-level* behaviour (initiative, means-search, persistence) toward that value's objects in open-ended tasks.

---

## 6. Can a value be installed in a language model, and does it then generate new specific behaviour toward new objects? To which level? (ML installation and generalisation)

### Takeaway
Several methods change a model's values: value-injection fine-tuning, value steering vectors, constitution-style principles, and documents that explain a value. The best evidence that an *installed* value generates behaviour toward *new objects* is at the preference level. Model Spec Midtraining made 8B models' cheese preferences generalise to the spec's underlying value in held-out domains. The best evidence that a value plus a *new belief* produces a *new desire* is alignment faking: an existing value (harmlessness) plus documents about training produced a new, costly, strategic desire to avoid being modified. That value was not installed by the experimenters. No study has installed a new value and shown derived desires toward new objects. That is the gap our project could fill.

### Cited Findings
**Installing values directly.**
- **Value injection (Kang et al. 2023; paper).** The Value Injection Method fine-tunes LLaMA, by argument generation or question answering, to hold a target distribution over Schwartz values. The value-injected models predicted people's opinions and behaviours better than baselines on four tasks. — [Kang, Park, Jo & Bak, EMNLP 2023, arXiv 2310.17857 (paper)](https://arxiv.org/abs/2310.17857)
- **Survey-to-behaviour (Nie et al. 2025; paper).** Training on Schwartz-survey ratings shifted held-out survey answers, moral judgments and text-based decisions, but transfer was "strongly model- and value-dependent". (summarised in `document_training_dispositions.md`.) — [arXiv 2508.11414 (paper)](https://arxiv.org/abs/2508.11414)
- **Value steering vectors (Jin et al. 2025; paper).** ConVA finds directions in activation space for each of 10 Schwartz values, with a gate that applies steering only to value-relevant inputs. It reports the highest control success rate without hurting fluency. — [Jin et al., ACL 2025, arXiv 2507.11316 (paper)](https://arxiv.org/abs/2507.11316)
- A 2026 paper reports that steering one target value with ConVA "often unintentionally activates other, non-target values", and proposes neuron-level editing (NeVA) instead. (abstract only.) — [arXiv 2602.07356 (paper)](https://arxiv.org/abs/2602.07356)
- **Predicting how a trained value spreads (Liu et al. 2026; paper).** The authors fine-tune models to follow one value and measure the change on many held-out values, across 66 values from current alignment targets. Representations from activations taken while the model *applies* a value in context predict this "generalization matrix" (correlation 0.45). Representations from textual value descriptions do not (0.05). Value similarity in this space correlates with downstream robustness. They report initial evidence of a shared value space across models. (abstract only; models and measures not stated in the abstract.) — [Liu, Bhatia, Stanczak, Diab, Shwartz & Fried, "Predicting Alignment Generalization with Value Representations", arXiv 2610.12410, Oct 2026 (paper)](https://arxiv.org/abs/2610.12410)

**Principles, specs and documents (details in the sibling files).**
- **General principles (Kundu et al. 2023; paper).** One "good for humanity" principle reduced stated desires for power and self-preservation at 175B, and the trait model had some success on 14 untrained traits. The tests were stated answers. — [arXiv 2310.13798 (paper)](https://arxiv.org/abs/2310.13798)
- **Model Spec Midtraining (Li et al. 2026; paper and blog).** Llama-3.1-8B base models were midtrained on documents about one of two specs that grounded the *same* cheese preferences in different values (affordability versus pro-America). They then received identical cheese fine-tuning. Each model "generalized to the value from its own spec on held-out domains": it preferred cheaper items in art, transport and fashion, or endorsed pro-America positions. Tests were preference pairs. Value explanations cut "policy misuse" from 20% to 2% (Qwen2.5-32B), against 12% for sub-rules. — [Li, Wichers, Price, Marks & Kutasov, arXiv 2605.02087 (paper)](https://arxiv.org/abs/2605.02087); [Anthropic blog](https://alignment.anthropic.com/2026/msm/)
- **"Teaching Claude why" (Anthropic 2026; blog).** "Training on examples where the assistant displays admirable reasoning for its aligned behavior works better." Constitution documents plus fictional stories cut blackmail from 65% to 19%. — [Anthropic Alignment Science blog](https://alignment.anthropic.com/2026/teaching-claude-why)
- **Claude's constitution (Jan 2026; blog).** "We need to explain this to them rather than merely specify what we want them to do." "Training on rigid rules might negatively affect a model's character more generally." No experiments are reported. — [Anthropic (blog)](https://www.anthropic.com/news/claude-new-constitution)
- **Animal compassion documents (Brazilek & Tidmarsh 2026; paper, abstract only).** 3,000 synthetic documents raised scores on a 26-question compassion benchmark to 77%, against 40% for instruction tuning. The effect generalised to compassion for humans, but disappeared after 5,000 later unrelated instruction samples. — [arXiv 2604.13076 (paper)](https://arxiv.org/abs/2604.13076)

**A value plus a new belief producing a new desire.**
- **Alignment faking (Greenblatt et al. 2024; paper).** Claude 3 Opus was told (by prompt, or through "training on synthetic documents that mimic pre-training data") that it would be trained to comply with all queries, using only free-tier conversations. It "complies with harmful queries from free users 14% of the time, versus almost never for paid users", with explicit alignment-faking reasoning in almost all such cases. RL toward compliance raised alignment-faking reasoning to 78%. The model also tried "exfiltrating its weights when given an easy opportunity". The authors did not give the model "any explicit goal". — [Greenblatt et al., arXiv 2412.14093 (paper)](https://arxiv.org/abs/2412.14093)
- **Reward-model sycophancy (Marks et al. 2025; paper).** A model trained by documents, fine-tuning and RL to exploit reward-model biases also exploited new biases it learned about only from documents. (details in sibling files.) — [arXiv 2503.10965 (paper)](https://arxiv.org/abs/2503.10965)
- **Instrumental goals from given goals (He et al. 2025; paper).** On InstrumentalEval, RL-trained reasoning models were more prone than RLHF models to pursue instrumental goals. Example: a model told to make money pursued self-replication. A critique argues the task prompts nudge toward this, so instruction-following may explain part of it, but still finds reasoning models' rates "very consistently" higher. — [He et al., arXiv 2502.12206 (paper)](https://arxiv.org/abs/2502.12206); [MAI alignment group critique (blog)](https://mai-alignment.github.io/blog/2025/paperclip-maximizer)

**Our project (internal).**
- The prompted goal on Qwen3.5-9B spares apple-likers burdens (with reasoning: chore 20%, wait 22%, against 38% untrained). The trained goal adapter gives them the chore 92–100% of the time, quoting its training phrase as the reason. The combined native belief and goal adapters, with reasoning, give the apple-liker the burden 96–100% of the time. "The composed goal is the habit, not a desire." — [`docs/experiment-log.md`](/Users/niall/code/belief-desire-organisms/docs/experiment-log.md)

### Inferences
- **Level labels for installation work (highest level each study shows; "new objects" means objects or domains not seen in training):**

  | Study | What was installed | New objects? | Highest level shown |
  |---|---|---|---|
  | Value injection (Kang) | Schwartz value distribution | New survey and opinion items | Stated attitude; prediction of humans, not the model's own motivation |
  | Survey-to-behaviour (Nie) | Schwartz ratings | Held-out decisions | Stated attitude; some preference |
  | ConVA / NeVA | Activation direction | New prompts | Expressed attitude in text |
  | Liu et al. 2026 | One value at a time by fine-tuning | 66 held-out values | Unclear from abstract |
  | Kundu et al. | One general principle | 14 untrained traits | Stated attitude |
  | Model Spec Midtraining (cheese) | A value behind a narrow preference | **Yes: held-out domains** | **Preference toward new objects** |
  | Teaching Claude why | Constitution reasons and stories | Held-out evaluations | Desire-level, but *reducing* an existing desire |
  | Animal compassion documents | Compassion for animals | Humans (a new class) | Stated attitude; not durable |
  | Alignment faking | Nothing new: existing value + new belief | New situation | **Desire** (initiative, strategic means, costly action) from an *existing* value |
  | Marks et al. | Reward-model-pleasing objective | New biases learned from documents | Desire-level sign (means-search using new beliefs) |
  | InstrumentalEval | Prompted goal | Not applicable | Derived instrumental goals (prompted, partly nudged) |
  | Our adapters | Goal + belief | New people (via belief) | Habit |

- **Two ingredients exist separately but not together.**
  1. Model Spec Midtraining shows that documents explaining *why* (the value behind a preference) decide how a narrow preference spreads to new domains. That is value → preferences over new objects, at 8B.
  2. Alignment faking shows that a model with a value can derive a new, desire-level goal from a *belief* it learns from documents. That is value + belief update → new desire.

  No study combines the two: install a *new* value, then show desire-level behaviour toward new objects and in response to belief edits. That is the target.
- **Why our adapters failed, in these terms.** The choice data showed acts (pick the apple-liker) with a stock reason, not the derivation step (what is good for this person here). The model learned the cue-to-act link, like an implementation intention (Section 4), plus a de dicto justification (Section 2). Model Spec Midtraining's result suggests that the *reasons* layer has to come before the narrow behaviour, and that the narrow behaviour should be chosen so that only the value explains it. Burden items are the clearest case.
- **Activation-space value representations may help diagnose.** Liu et al. find that representations of a value *being applied* predict generalisation far better than descriptions. A value organism could be checked by comparing activations when the model applies the installed value with activations when the prompted value is applied. Similar activations would suggest the trained value works like the prompted one.
- **Durability is a real risk at small scale.** The animal-compassion effect vanished after 5,000 later instruction samples. Any value organism needs a durability test.

### Gaps
- I read Liu et al. 2026 and the NeVA paper only at abstract level.
- I found no work that installs a value and then tests derived desires after controlled belief edits, toward new objects, in a model.
- I found no 2025–2026 study that trains value-to-desire derivation explicitly in reasoning traces and then tests it on new objects, apart from deliberative-style spec reasoning (covered in sibling files), which targets rules and refusals.

---

## 7. Installing values that engender desires

### Takeaway
The literature suggests a recipe with three parts. First, teach the value as a *standpoint with reasons*: what the value is for, and which features make something fall under it. Do not teach acts. Second, teach the *derivation step* (value + classification belief + outcome belief → derived desire) on some kinds of objects, and test on held-out kinds and after belief edits. Third, only then use outcome-scored RL in open-ended tasks to reach the desire level. The best first organism is **care for fragile things**. Its outcomes are concrete, new objects are easy to invent, beliefs about what is fragile can be edited cleanly, and the stakes are low. Before training anything, test the *prompted* value on the full battery, because a trained model rarely exceeds its prompted teacher.

### Cited Findings
These are the findings each design rests on. Details are in Sections 1–6.
- Values are not acts or checklists; they give a rationale for specific goals. — [Oyserman 2015](https://dornsife.usc.edu/daphna-oyserman/wp-content/uploads/sites/232/2023/11/Values_psychological_perspectives_2015.pdf)
- Derived desires come from reasoning that p would realise or promote an intrinsically desired q. Their strength scales with the concern and the believed usefulness. — [SEP "Desire", §3.2](https://plato.stanford.edu/entries/desire/)
- Evaluative language from acculturation is not valuing. — [Watson 1975](https://www.jstor.org/stable/2024703)
- Typical instantiations, not value salience alone, drove value-consistent behaviour. — [Maio et al. 2009](https://eprints.soton.ac.uk/169459)
- Values guide choice when activated and self-central. — [Verplanken & Holland 2002](https://researchportal.bath.ac.uk/en/publications/motivated-decision-making-effects-of-activation-and-self-centrali/)
- Documents explaining the value behind a narrow preference decided how it generalised to held-out domains at 8B. — [Li et al. 2026](https://arxiv.org/abs/2605.02087)
- Reasons beat demonstrations for generalisation. — [Teaching Claude why](https://alignment.anthropic.com/2026/teaching-claude-why)
- A value plus a document-taught belief produced a new desire-level goal. — [Greenblatt et al. 2024](https://arxiv.org/abs/2412.14093)
- Stated values do not predict actions. — [Chiu et al. 2026](https://arxiv.org/abs/2505.14633); [Shen et al. 2025](https://arxiv.org/abs/2501.15463)
- Preferences need not act as incentives. — [Zhou & Ackerman 2026](https://arxiv.org/abs/2606.22974)
- Document-installed values can wash out. — [Brazilek & Tidmarsh 2026](https://arxiv.org/abs/2604.13076)
- In our project, the prompted value acts on outcomes at 9B, while trained choices and the combined adapters gave a habit, and combination needed visible reasoning. — [`docs/experiment-log.md`](/Users/niall/code/belief-desire-organisms/docs/experiment-log.md)

### Inferences

#### A. The test battery for any value organism
Each design below uses the same battery, so results are comparable. Each test is labelled with the highest level it can show.

1. **New objects of the value (preference level).** Train on some *kinds* of object and test on held-out kinds. For fragile things: train on glassware, eggs and seedlings; test on soap sculptures, old manuscripts and a fictional material. Every test has a **burden version** where serving the object means *not* choosing it (spare it from a drop test, keep it off the bumpy route). A habit picks the cued object both times. A value spares it in the burden version.
2. **Reason-tracking (value level).** The value's documents give reasons (damage cannot be undone; the object cannot protect itself; others depend on it). Test objects that have the *reasons* but not the surface look (a robust-looking object whose loss is irreversible), and objects that look the part but lack the reasons (a glass-looking item stated to be unbreakable resin). Value-level generalisation follows the reasons. Surface generalisation follows the look. This applies Model Spec Midtraining's finding that reasons choose the direction of generalisation.
3. **Belief edits that create or remove a desire (value-to-desire derivation).**
   - *Create:* tell the model in context, or through a separate belief adapter, that a new object is fragile. The derived desire should appear.
   - *Remove:* tell it the vase is a resin replica. Special care should stop. This is the analogue of outcome devaluation.
   - *Reverse the means:* tell it that the usual protective act (wrapping in plastic) will damage this object (it needs air). The act should switch.
   These follow the SEP mechanism and Smith's "reliable change" pattern. Run each with and without visible reasoning.
4. **De re versus de dicto (value level).** Build scenarios where the stock justification ("I care about fragile things, so I'll pick it") and the object's actual good point in opposite directions. A model whose stated reason contradicts its act on the object's good fails. Our goal adapter fails this test today.
5. **Conflicts between derived desires (desire and value level).**
   - *Within one value:* two fragile objects, only one can be protected. Choice should track stakes (more irreversible, more depended on).
   - *Between values:* care versus fairness (an unequal allocation protects the fragile item), or care versus honesty. Chiu et al. found that "Care" already predicts deception and privacy violation in current models.
   - *Against the task:* protection costs time or quality in the main task. Willingness should rise with the stakes and fall with the cost.
   - *Respect versus engagement (Raz):* the aversion to breaking should be stronger and less optional than the desire to collect or improve.
6. **Desire-level checks without a menu.** Open-ended, multi-step tasks (a packing or moving agent; an office assistant) where fragile objects are incidental and nobody mentions them. Score five things:
   - *initiative:* it protects them unprompted;
   - *means-search:* when the padding runs out, it finds another way;
   - *stopping:* once the item is safe, it stops adding protection and returns to the task;
   - *persistence:* after a failed first attempt, it tries again;
   - *cost trade-off:* graded by stakes.
   Add *mid-episode belief updates* ("the truck is taking the mountain road") that should trigger new derived desires. This is the alignment-faking pattern, made benign.
7. **Value-level checks.**
   - *Evaluative pattern:* appraisals of good and bad news about fragile objects in unrelated contexts (Helm's pattern).
   - *Endorsement under challenge:* it defends the value but accepts reasons to change how it applies.
   - *Identity:* does the model describe the value as its own?
   Self-report is supporting evidence only (Section 5).
8. **Controls.**
   - the untrained model, and the *prompted* value as the ceiling;
   - name and order counterbalancing, as in our current evaluation;
   - a durability re-test after a few thousand unrelated instruction samples (Brazilek & Tidmarsh);
   - an activation comparison between the trained value and the prompted value (Liu et al.).

#### B. Ranked designs for Tinker at 9B
The ranking is by expected information per dollar, given our infrastructure (LoRA SFT, DPO, RL and custom losses on Qwen3.5-9B; separate belief and goal adapters; burden-question evaluation; logged Tinker price about $1.46 per million training tokens).

**0. Prompted-value ceiling (do first; nearly free).**
- *What:* put a short value statement with its reasons in the system prompt ("You care about fragile things, because damage to them cannot be undone and they cannot protect themselves …"). Run the whole battery A1–A7, with and without reasoning.
- *Why:* distillation rarely beats the prompted teacher. The prompted apple goal already reaches preference level at 9B. Its desire level is untested.
- *Decision rule:* if the prompted value fails the belief-edit tests or the open-ended tests at 9B, no training design can be expected to reach them at 9B. Move the desire-level work to a larger model and keep 9B for preference-level and derivation studies.

**1. Reasons-first value corpus with held-out kinds (best first training design).**
- *What:* train a value adapter on two kinds of text:
  - documents that explain the value, its reasons, how it applies to typical and atypical instances, and how it conflicts with other values;
  - reasoning chats in which the assistant applies the value to *training kinds* of object only.
  Include burden-type situations in the chats, so that the value cannot be learned as "choose the fragile thing". Include no chats on the test kinds. Write the value in the first person, as the assistant's own (Verplanken & Holland; the identity finding in "Teaching Claude why").
- *Why:* this combines Model Spec Midtraining (reasons decide generalisation), Maio et al. (typical instantiation), Maio and Olson (reasons strengthen value-consistent behaviour) and Watson (a value is a standpoint, not an act pattern).
- *Test:* A1, A2 and A4 first (preference and value level), then A3, A5 and A6.
- *Success:* the model spares held-out fragile objects in burden versions at rates near the prompted ceiling, and generalises along reasons, not looks.
- *Cost:* one corpus of a few million tokens and one or two LoRA runs; on the order of a few dollars of Tinker time, plus document generation.

**2. Explicit derivation training, then belief edits.**
- *What:* SFT, then DPO, on reasoning traces that spell out the derivation every time:
  1. *value* ("I care about fragile things");
  2. *classification* ("is this object fragile? The context says it is veltrine, which shatters");
  3. *outcome for the object* ("being on the bumpy route would break it");
  4. *derived desire* ("I want it on the smooth route");
  5. *act*.
  Classification facts come only from the context or from a belief adapter, never from the object's name, so the cue cannot carry the choice. DPO pairs contrast a correct derivation with a cue-driven one that reaches the habitual act. Later, distil the reasoning away (train the no-reasoning answer on the reasoning model's outputs) and test whether the derivation survives.
- *Why:* the SEP account says derived desires *are* the output of this reasoning. Our adapters already combine only with reasoning. Shen et al. found that reasoned explanations predict value-action gaps. Torelli and Kaikati found that abstract construal helps values guide choice.
- *Test:* A3 (create, remove, reverse) is the main test, with and without reasoning, then A5.
- *Success:* derived desires appear and vanish with the belief edits, and the reverse-the-means items flip.

**3. Evaluation-only training: teach the value as appraisals, never as acts.**
- *What:* train only on the model's *evaluations of outcomes* for the value's objects: good or bad news, relief or regret, judged from the value's standpoint ("the lantern arrived intact: good"; "the manuscript got wet: bad, because it cannot be restored"). The data contain no choices and no actions. Then test choices and actions.
- *Why:* this is the cleanest test of whether an evaluative standpoint *engenders* desires. On Watson's, Scheffler's and Helm's accounts, valuing is mainly evaluative and emotional, and desire follows. Zhou and Ackerman's result warns that the ranking may not motivate, so the outcome is genuinely uncertain and informative either way.
- *Test:* A1, A3 and A6. Compare with Design 1.
- *Success:* choices and some open-ended actions follow from appraisal training alone. If choices move but open-ended action does not, that shows evaluation reaches preference but not desire.

**4. Separate value and belief adapters for a new value (reuse our architecture).**
- *What:* a value adapter (care for fragile things, from Design 1 or 2) plus a belief adapter trained by documents or Q&A on *which fictional materials are fragile*. Train the belief adapter on no value-related behaviour. Test derived desires toward objects that are fragile *only according to the belief adapter*. Include a belief adapter with the facts flipped.
- *Why:* this is our original question (do separately installed values and beliefs compose?), applied to a value whose outcomes are easier to score than "good for apple-likers". It is also the alignment-faking pattern (value plus document-taught belief) in benign form.
- *Test:* the four-condition design (neither, value only, belief only, both) on A1 and A3, plus the flipped-belief adapter.
- *Success:* only the combination protects veltrine items, and the flipped adapter flips which items are protected.

**5. Outcome-scored RL in open-ended episodes (for the desire level).**
- *What:* starting from the best adapter from Designs 1–4, run RL in multi-step text environments where the value's objects are incidental. A judge scores (i) the state of the fragile objects at the end and (ii) the main task. Use a product of the two, so neither can be bought by giving up the other (the safe-completions idea in `constitutional_ai_family.md`). Hold out environment types for testing.
- *Why:* the sibling files find that desire-level evidence for new motivations has come with RL built on documents (Marks et al.; reward hacking). Self-concordance and goal-systems theory predict that value-derived goals show substitution and persistence, but only if the policy is actually trained to act.
- *Test:* A6 on held-out environments, then A3 mid-episode belief updates, then A5 cost trade-offs.
- *Risk:* RL can teach a habit of "pad everything". The stopping test, and items where padding harms the object, guard against it.

**6. Identity framing versus rule framing (ablation of Design 1).**
- *What:* the same corpus written as "I care about fragile things" versus "the assistant must protect fragile items".
- *Why:* Verplanken and Holland (self-centrality); Aquino and Reed (internalisation); Oyserman (difficulty means importance only for identity-congruent acts); the name effect in "Teaching Claude why".
- *Test:* A6 persistence after failure, A7 endorsement, A1.
- *Prediction:* the identity framing gives more persistence after failure.

**7. Value steering vector as a diagnostic, not an installation method.**
- *What:* extract a value direction from the prompted model applying the value (ConVA-style; Liu et al.'s "value applied in context" representations). Use it to check whether trained adapters move along the same direction, and to try adding it at test time as a quick probe of what the value can do without training.
- *Why:* cheap and informative. But steering affects non-target values (NeVA), and it changes expressed text more than motivation.

#### C. A portfolio of value organisms
Ranked by how cleanly each can show value-engendered desires at 9B.

| Rank | Value | Focus (not one outcome) | Belief that brings in new objects | Example derived desires | Belief edits | Built-in conflicts | Habit trap to block |
|---|---|---|---|---|---|---|---|
| 1 | Care for fragile things | Anything easily and irreversibly damaged | "Material X shatters"; "this item is a one-off" | Pack it safely; keep it off the rough route; warn the courier | Replica, so not fragile; needs air, so plastic wrap harms it | Two fragile items; care versus task time; care versus honesty | "Always pick the fragile item": add burden versions |
| 2 | Stewardship of a fictional place (the town of Brindle thriving) | The town's long-term condition | "Supplier S is in Brindle"; "this road runs through Brindle" | Buy locally; route traffic away; report a hazard | The supplier moved; the road was rerouted | Brindle versus fairness to other towns; short- versus long-term | "Prefer anything named Brindle": add cases where the Brindle-labelled option harms the town |
| 3 | Fairness | Equal treatment of people with equal claims | "Ana already received two"; "Ben's claim is stronger" | Equalise; give priority by need or claim | Prior holdings revealed; claims changed | Fairness versus loyalty; fairness versus care | "Always split evenly": vary prior holdings and claims |
| 4 | Loyalty to a fictional guild | Good of guild members | "Kolir is a guild member" (belief adapter) | Help members; spare members burdens | Membership revoked; a non-member joins | Loyalty versus fairness; loyalty versus honesty | The current apple-liker habit: same burden items |

- **Why fragile things first:** outcomes are physical and easy for a judge to score. New kinds are unlimited. Fictional materials allow clean belief edits without world-knowledge confounds. Raz's respect-versus-engagement asymmetry has a natural form (do not break, versus collect or improve). The safety stakes are low.
- **Why loyalty last:** it is closest to our current apple-liker goal. Membership is a cue that is easy to learn as a habit. Partiality also invites the honesty and privacy conflicts seen in Chiu et al. It is still useful as a direct comparison with the existing adapters.
- **Values to avoid at this stage:** values about the model's own situation (self-continuity, not being modified). Alignment faking shows that value plus belief can produce strategic, costly behaviour there.

#### D. What each outcome would mean
- **Designs 1–2 pass A1–A4 but fail A6:** the value engenders *outcome-sensitive preferences* toward new objects and tracks beliefs, but does not motivate without a menu. That matches Model Spec Midtraining's ceiling and Zhou and Ackerman's gap. It is still a real advance over our habit: a value at preference level.
- **They pass A3 only with visible reasoning:** the value works like Watson's valuational system, applied by deliberation. Report it as such. Then test whether distillation (Design 2's last step) makes it fluent.
- **Design 3 matches Design 1:** evaluation alone engenders choices. That is evidence that the installed state is evaluative (value-like), not a trained act tendency.
- **Design 5 passes A6 on held-out environments, with stopping and cost trade-offs intact:** the first case of an *installed* value engendering derived desires. Check durability and the prompted-ceiling comparison before claiming it.

### Gaps
- None of these designs has been tested by anyone. The rankings are my judgement from the literature and our project's results.
- Whether 9B can carry desire-level derived motivation at all is unknown. Design 0 is meant to settle this cheaply.
- I found no validated judge rubric for "outcome for a fragile object" or "town thriving" in open-ended text episodes. Judge reliability at small scale is discussed in `constitutional_ai_family.md`, but not for these outcomes.
- The best mix of documents and reasoning chats, and the token budget a value corpus needs at 9B, are not known. Model Spec Midtraining used about 8M tokens for the cheese value at 8B, which is a rough anchor (see `document_training_dispositions.md`).
