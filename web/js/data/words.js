// Word list for eye-typing predictions, roughly in order of how common the
// words are (everyday English plus words that matter in care situations).

export const WORDS = (
  'i you the to a and is it yes no please thank thanks help need want can what my me that ' +
  'in of for be have not this are do on with we he she they your how now so but was at go ' +
  'am will like just know get feel good okay ok hello hi bye water time here there when where ' +
  'why who more some all one about up out if an or as by from them then than too very well ' +
  'love would could should did does don\'t can\'t i\'m it\'s that\'s let\'s think see look come make ' +
  'take give tell talk say said call back home today tomorrow tonight morning night later soon ' +
  'again still also maybe really sorry nice great fine bad better best tired hungry thirsty cold ' +
  'hot warm pain hurt hurts sick doctor nurse medicine bathroom toilet bed sleep rest wake chair ' +
  'window door light lights tv music phone computer book read watch listen family friend friends ' +
  'mom mum dad mother father wife husband son daughter brother sister baby child children people ' +
  'person man woman name day week month year hour minute minutes second please stop start wait ' +
  'turn move sit stand lie down open close off on left right side head eyes face mouth nose ' +
  'arm arms hand hands leg legs back neck chest stomach feet foot skin itch itchy breathe breath ' +
  'eat drink food coffee tea juice milk soup bread breakfast lunch dinner snack sugar salt more less ' +
  'enough much many little big small new old long short first last next other same different ' +
  'happy sad angry scared worried bored calm comfortable uncomfortable funny beautiful lovely ' +
  'thank you please yes no maybe later now here there this that these those something nothing ' +
  'everything anything someone everyone anyone nobody always never sometimes often usually ' +
  'because until while after before during again already almost enough together alone outside ' +
  'inside garden walk drive car bus shop shopping money work school play game games fun question ' +
  'answer word words letter message email write send help understand remember forget believe hope ' +
  'wish miss need needs wants use used find found keep kept put bring brought leave left lose ' +
  'win try trying learn show showed hear heard ask asked answer done finished ready busy free ' +
  'early late fast slow quiet loud clean dirty wet dry heavy easy hard difficult right wrong true ' +
  'sure important possible problem idea thing things place world life house room kitchen table ' +
  'glass cup plate spoon fork knife pillow blanket clothes shirt shoes socks hair teeth glasses ' +
  'appointment visit visitors birthday holiday weekend monday tuesday wednesday thursday friday ' +
  'saturday sunday january february march april may june july august september october november ' +
  'december weather rain sun snow wind outside cold hot news story film movie song picture photo'
).split(/\s+/).filter((w, i, all) => w && all.indexOf(w) === i);

// Likely first words when nothing has been typed yet.
export const STARTERS = ['I', 'Please', 'Can', 'Thank'];
