"""
Builds the one combined dataset that every functionality in this project reads.

WHAT: writes data/processed/combined_dataset.csv and data/processed/senders.csv.
WHY:  no public dataset has Hinglish scam text AND sender-network information, so we
      combine what is real with what we have to generate ourselves.
HOW:  run `python -m src.data_builder`.

The four ingredients, and how honest each one is
------------------------------------------------
1. REAL   - UCI SMS Spam Collection: 5,574 real English messages (747 spam / 4,827 ham).
2. REAL   - Hugging Face Hinglish seed: 87 hand-written rows (47 scam / 40 benign).
            CAUTION: that HF dataset is 97.7% a re-upload of UCI. We measured this and
            drop every row that already appears in UCI, keeping only the 87 new ones.
            Without that step, 3,700 duplicate texts would land in the pool and the
            same message could appear in both train and test - textbook leakage.
3. SYNTHETIC - ~2,500 Hinglish messages we generate from templates (9 scam families,
            6 benign families) with Romanized spelling variants.
4. SYNTHETIC - the whole sender->recipient network. Stated as a limitation in
            docs/PAPER_NOTES.md. Graph results show the METHOD works on structure of
            this kind; they are not evidence about real telecom traffic.

Output columns
--------------
message_id, text, sender_id, recipient_id, timestamp, label, language_mix, source
plus these helper columns, which the splits and F3 need:
template_family, sender_ring, split_leaksafe, split_time
"""

import io
import urllib.request
import zipfile

import numpy as np
import pandas as pd

from src.config import (
    ACTIVITY_SPREAD,
    BURST_HOURS_MAX,
    BURST_HOURS_MIN,
    COMBINED_CSV,
    BENIGN_CONTACTS,
    FRAC_BENIGN_SENDERS_LOOK_ODD,
    RING_VICTIM_POOL,
    SOCIAL_CIRCLE,
    FRAC_SCAM_SENDERS_LOOK_NORMAL,
    HF_HINGLISH_DATASET,
    MESSAGES_PER_BENIGN_SENDER,
    MESSAGES_PER_RING,
    MESSAGES_PER_SCAM_SENDER,
    N_RECIPIENTS,
    N_SYNTHETIC_MESSAGES,
    PROCESSED_DIR,
    RANDOM_SEED,
    RAW_DIR,
    UCI_SMS_URL,
)

# ---------------------------------------------------------------------------
# Scam message templates: 9 families, the scams that actually circulate in India.
# {curly} parts are filled in with random values by fill_placeholders().
# ---------------------------------------------------------------------------
SCAM_TEMPLATES = {
    "kyc_expiry": [
        "Dear customer, aapka {bank} KYC {hours} ghante me expire ho jayega. Turant verify karein: {link}",
        "{bank} alert: KYC pending hai, account band ho jayega. Update karo abhi: {link}",
        "URGENT {bank}: Your KYC is incomplete. Khata band hone se bachane ke liye {link} par jaayein.",
        "Aapka {bank} account KYC ke bina {hours} hours me deactivate. Verify now {link}",
    ],
    "upi_refund": [
        "{amount} aapke account se galti se kat gaya hai. Refund ke liye UPI PIN daalein: {link}",
        "UPI refund of {amount} pending hai. Claim karne ke liye turant click karein {link}",
        "Dear user, {amount} refund failed. Bank details verify karo yahan: {link}",
        "{bank} refund {amount} ka process karne ke liye is link par UPI pin enter karein: {link}",
    ],
    "otp_share": [
        "{bank} fraud department se call. Aapka OTP {otp} humein batayein warna account band.",
        "Security check: apna OTP aur PIN share karein support ke saath, warna khata block ho jayega.",
        "Aapke account me suspicious transaction hai. OTP {otp} reply karein turant.",
        "To stop this transaction, share the OTP sent to you with our agent immediately.",
    ],
    "lottery_prize": [
        "Congratulations! Aapko {amount} ka lottery inaam mila hai. Claim ke liye {link}",
        "Lucky draw winner! {amount} jeeta hai aapne. Processing fee bhejein is UPI par: {upi}",
        "Aapka mobile number {amount} ke lucky draw me select hua hai. Turant claim karein {link}",
        "You won a free iPhone! Shipping charge {amount} pay karein: {upi}",
    ],
    "loan_fee": [
        "Instant loan {amount} approved! Bina documents. Registration fee bhejein {upi}",
        "Paisa loan chahiye? 10 minute me approval, sirf processing fee {amount}. Apply {link}",
        "Pre-approved personal loan {amount}. File charge {amount2} pay karein yahan: {upi}",
        "Loan sanction ho gaya hai. Insurance fee {amount} jaldi bhejein warna cancel.",
    ],
    "electricity_disconnect": [
        "Bijli connection aaj raat kat jayega. Bill pay karein turant: {link}",
        "FINAL NOTICE: Your electricity will be disconnected in {hours} hours. Pay {amount}: {link}",
        "Electricity bill {amount} pending hai, meter band ho jayega. Pay karo {link}",
        "Bijli vibhag se: bill update nahi hua. Disconnection rokne ke liye {link} par jaayein.",
    ],
    "parcel_customs": [
        "Aapka parcel customs me ruka hua hai. Clearance fee {amount} pay karein: {link}",
        "Courier delivery failed. Address update aur {amount} fee ke liye click {link}",
        "Your package is held. Customs duty {amount} pay karo warna parcel return ho jayega: {link}",
        "Parcel delivery pending hai. {amount} ka charge bhejein is UPI par: {upi}",
    ],
    "job_fee": [
        "Work from home job! {amount} per month. Registration fee {amount2} bhejein: {upi}",
        "Job alert: data entry, {amount} salary, no experience. Fee {amount2} pay karein {link}",
        "Aapka job selection ho gaya hai. Security deposit {amount2} jaldi bhejein warna cancel.",
        "Part time job {amount} monthly. Joining ke liye {amount2} ka payment karein: {upi}",
    ],
    "family_emergency": [
        "Beta, mummy ko hospital me admit karaya hai. {amount} turant bhej do is account par.",
        "Papa ka accident ho gaya hai, emergency me {amount} chahiye. Kisi ko mat batana.",
        "Your son is in police custody. Send {amount} immediately. Do not call anyone.",
        "Bhai main phone kho gaya, naya number hai. {amount} urgently bhej de, baad me wapas.",
    ],
}

# ---------------------------------------------------------------------------
# Benign message templates: 6 families of perfectly normal messages.
# These deliberately include an OTP message and a bank message with NO link, because
# a detector that flags every message containing "OTP" or "bank" is useless.
# ---------------------------------------------------------------------------
BENIGN_TEMPLATES = {
    "delivery_update": [
        "Aapka {item} order deliver ho gaya hai. Track ID {trackid}: {safelink}",
        "Your {item} parcel will arrive in {hours} hours. Track here: {safelink}",
        "{name}, Flipkart order shipped hai, kal tak aa jayega. Track: {trackid}",
        "Delivery agent {hours} minute me pahunch raha hai {place} par. Please ghar par rahein.",
        "Your {item} order is out for delivery. Order ID {trackid}, details at {safelink}",
    ],
    "otp_genuine": [
        "{otp} is your OTP for {bank} login. Kisi ke saath share na karein.",
        "Your verification code is {otp}. Valid for {hours} minutes. Do not share with anyone.",
        "{bank}: OTP {otp} for your transaction of {amount}. Bank kabhi OTP nahi maangta.",
        "Use {otp} to log in {name}. Agar aapne request nahi ki to ignore karein.",
    ],
    "bank_statement": [
        "{amount} credited to your {bank} account. Available balance updated.",
        "{bank}: {amount} debited for electricity bill. Payment successful raha.",
        "Aapka {bank} monthly statement ready hai: {safelink}",
        "Payment of {amount} received from {name}. UPI Ref {trackid}",
        "{bank}: {amount} debited. Statement dekhein {safelink}",
    ],
    "friend_chat": [
        "{name} yaar kal {item} lene chalein? {time} baje nikalte hain.",
        "Match dekhne aa raha hai na Sunday ko? {place} par milte hain {time} baje.",
        "Bhai woh notes bhej de WhatsApp par, kal exam hai {name}.",
        "Lunch kiya {name}? Main {place} se nikal raha hoon {time} baje.",
    ],
    "family_message": [
        "Beta khana kha liya? {time} baje call karna, mummy baat karegi.",
        "Market se {item} le aana aate waqt {name}.",
        "Papa ki tabiyat theek hai ab, doctor ne {hours} din rest bola hai.",
        "{name} ki shaadi December me hai, sabko invite bhej diya hai.",
    ],
    "promo_optin": [
        "{name}, aapke {item} order par 10% off. Offer dekhein: {shortlink}",
        "Thank you for shopping! Rate your {item} delivery here: {shortlink}",
        "{bank} rewards: aapke {amount} points ready hain. Details: {shortlink}",
        "Your {item} is back in stock. Shop now: {shortlink}",
    ],
    "work_message": [
        "Meeting {time} baje reschedule ho gayi hai {place} me. Calendar update kar lein.",
        "Report attach kar di hai email me. {name} review karke batayein.",
        "Project tracker update ho gaya hai, dekh lein: {safelink}",
        "Team lunch Friday ko hai {place} par. RSVP kar dein {name}.",
        "Project files bhej diye hain, {time} tak feedback dena {name}.",
    ],
}

# ---------------------------------------------------------------------------
# Romanized spelling variants.
# There is no official way to spell a Hindi word in English letters, so real messages
# spell the same word many ways. This is the whole reason F6 (phonetic codes) exists.
# ---------------------------------------------------------------------------
SPELLING_VARIANTS = {
    "kyc": ["kyc", "keyc", "k y c", "KYC"],
    "turant": ["turant", "turrant", "turat", "turnat"],
    "khata": ["khata", "khaata", "khatha"],
    "band": ["band", "bnd", "bandh"],
    "jaldi": ["jaldi", "jaldee", "jldi"],
    "paise": ["paise", "paisa", "pese"],
    "account": ["account", "acount", "a/c", "acc"],
    "verify": ["verify", "varify", "verifiy"],
    "bhejein": ["bhejein", "bhejien", "bhej dein"],
    "karein": ["karein", "karen", "kare"],
}

# Values used to fill the {curly} placeholders in the templates above.
PLACEHOLDER_VALUES = {
    "bank": ["SBI", "HDFC", "ICICI", "Axis Bank", "PNB", "Kotak"],
    "amount": ["Rs 5000", "Rs 10,000", "Rs 2,500", "Rs 50,000", "Rs 1,00,000", "Rs 999"],
    "amount2": ["Rs 500", "Rs 1,000", "Rs 250", "Rs 2,000"],
    "hours": ["2", "6", "12", "24", "48"],
    "otp": ["482913", "100293", "774512", "839201"],
    "trackid": ["FD31289045", "OD204871", "412398765123", "AWB9928371"],
    "link": [
        "http://sbi-kyc-verify.xyz", "https://hdfc-secure-update.in", "http://bit.ly/3xKyc9",
        "https://upi-refund-help.com", "http://icici.co.in-verify.xyz", "https://tinyurl.com/kycfix",
    ],
    "upi": ["help@ybl", "refund@paytm", "support2026@okaxis", "claim@ibl"],
    # LEGITIMATE links, for benign messages. These matter more than they look: if only
    # scam messages ever contained a URL, "has a link" would identify a scam almost
    # perfectly and every result in the paper would be inflated. Real delivery and bank
    # messages carry links, so ours must too. Note these use real company domains with
    # ordinary TLDs, so F2's link-risk features (shortener? throwaway TLD?) still tell
    # the two kinds of link apart - which is the signal we actually want to learn.
    "safelink": [
        "https://www.flipkart.com/orders", "https://www.amazon.in/orders",
        "https://www.sbi.co.in/netbanking", "https://www.icicibank.com/statement",
        "https://www.hdfcbank.com/login", "https://track.delhivery.com",
    ],
    # Shortened links in GENUINE messages. Legitimate brands really do send
    # "track your order: bit.ly/..." promotional SMS. We include them deliberately,
    # because otherwise a shortened link would mean "scam" with 100% certainty: F1's
    # LinkRisk node would jump straight to probability 1.0, and F7's counterfactuals
    # would collapse to "remove the link" every single time. These rows are the HARD
    # benign cases, and a detector worth reporting has to cope with them.
    "shortlink": ["bit.ly/3FlipKart", "tinyurl.com/myorder22", "bit.ly/SBIstmt"],
    # These exist so BENIGN templates vary as much as scam ones. Without them, a line
    # like "Lunch kiya?" generates the identical string every time, drop_duplicates
    # removes all but one, and we end up with almost no benign Hinglish to train on.
    "name": ["Rahul", "Priya", "Amit", "Sneha", "Kartik", "Neha", "Rohan", "Anjali",
             "Vikas", "Pooja", "Arjun", "Divya"],
    "item": ["doodh", "bread", "sabzi", "laptop", "phone cover", "books", "shoes", "kurta"],
    "time": ["9", "10", "11", "4", "5", "6", "7", "8"],
    "place": ["office", "ghar", "cafe", "station", "mall", "ground", "college"],
}


def download_uci_sms():
    """
    Download the UCI SMS Spam Collection zip into data/raw/ if it is not there yet.

    WHAT: fetches ~200 KB from archive.ics.uci.edu and returns the local zip path.
    WHY:  this is our only source of REAL messages at scale (5,574 of them).
    NOTE: data/raw/ is gitignored, so each person downloads it once themselves.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = RAW_DIR / "sms_spam_collection.zip"
    if not zip_path.exists():                         # don't re-download every run
        print(f"  downloading UCI SMS Spam Collection -> {zip_path}")
        urllib.request.urlretrieve(UCI_SMS_URL, zip_path)
    return zip_path


def load_uci_sms():
    """
    Read the UCI zip into a DataFrame with our own column names.

    WHAT: returns columns text, label, language_mix, source, template_family.
    WHY:  every source has to come out in the same shape so we can stack them.
    FORMAT: the file inside the zip is "label<TAB>text", one message per line.
    """
    with zipfile.ZipFile(download_uci_sms()) as archive:
        raw_bytes = archive.read("SMSSpamCollection")
    # errors="replace" because a handful of lines contain invalid byte sequences.
    text = raw_bytes.decode("utf-8", errors="replace")
    rows = [line.split("\t", 1) for line in text.strip().split("\n")]
    frame = pd.DataFrame(rows, columns=["uci_label", "text"])
    frame["label"] = (frame["uci_label"] == "spam").astype(int)   # spam -> 1, ham -> 0
    frame["language_mix"] = "english"
    frame["source"] = "uci_sms"
    # UCI has no templates, so family is just the label. The split explains why.
    frame["template_family"] = np.where(frame["label"] == 1, "uci_spam", "uci_ham")
    return frame[["text", "label", "language_mix", "source", "template_family"]]


def download_hinglish_seed():
    """
    Download the small real Hinglish dataset from Hugging Face.

    WHAT: returns the local path to data.json, or None if the Hub is unreachable.
    WHY:  these are the only REAL Hinglish rows we have. The project must still build
          without them (offline, or if the dataset disappears), hence the None path.
    """
    from huggingface_hub import hf_hub_download     # imported here so the rest works offline
    try:
        return hf_hub_download(
            HF_HINGLISH_DATASET, "data.json", repo_type="dataset",
            local_dir=str(RAW_DIR / "hf_hinglish"),
        )
    except Exception as error:                      # no network, renamed repo, rate limit...
        print(f"  WARNING: could not download Hinglish seed ({type(error).__name__}). Skipping.")
        return None


def load_hinglish_seed(uci_texts):
    """
    Read the Hugging Face seed and keep ONLY rows that are not already in UCI.

    WHAT: returns ~87 genuinely new rows (47 scam / 40 benign).
    WHY:  we measured that this dataset is 97.7% a re-upload of UCI (3,700 of 3,787
          rows are identical). Keeping them would mean the same text appears in both
          the train and test split, which inflates every score. So we deduplicate.
    ARGS: uci_texts - a set of the UCI message strings to compare against.
    """
    path = download_hinglish_seed()
    if path is None:
        return pd.DataFrame(columns=["text", "label", "language_mix", "source", "template_family"])
    # The json is one column called "rows", where each cell is a {"text":..,"label":..} dict.
    nested = pd.read_json(path)
    frame = pd.DataFrame(list(nested["rows"]))
    frame["text"] = frame["text"].astype(str).str.strip()
    frame = frame[~frame["text"].isin(uci_texts)].copy()          # <-- the dedup step
    print(f"  Hinglish seed: kept {len(frame)} rows not present in UCI")
    frame["language_mix"] = "hinglish"
    frame["source"] = "hf_hinglish_seed"
    frame["template_family"] = np.where(frame["label"] == 1, "seed_scam", "seed_benign")
    return frame[["text", "label", "language_mix", "source", "template_family"]]


def fill_placeholders(template, rng):
    """
    Replace every {placeholder} in a template with a random realistic value.

    WHAT: "KYC expire in {hours}h: {link}" -> "KYC expire in 24h: http://sbi-kyc.xyz"
    WHY:  so 4 templates can produce hundreds of different-looking messages.
    """
    text = template
    for name, choices in PLACEHOLDER_VALUES.items():
        marker = "{" + name + "}"
        while marker in text:                         # a template may use {amount} twice
            text = text.replace(marker, str(rng.choice(choices)), 1)
    return text


def apply_spelling_variants(text, rng):
    """
    Randomly respell some Romanized Hindi words, the way real senders do.

    WHAT: "turant verify karein" -> "turrant varify karen"
    WHY:  real Hinglish has no fixed spelling. If every generated message spelled
          "turant" the same way, our model would look far better than it deserves and
          F6 (the phonetic layer) would have nothing to prove.
    """
    words = text.split()
    for index, word in enumerate(words):
        key = word.lower().strip(".,!:")
        if key in SPELLING_VARIANTS and rng.random() < 0.5:    # respell half the time
            words[index] = str(rng.choice(SPELLING_VARIANTS[key]))
    return " ".join(words)


def generate_synthetic_messages(n_messages, rng):
    """
    Generate n Hinglish messages from the template families, half scam, half benign.

    WHAT: returns columns text, label, language_mix, source, template_family.
    WHY:  this is the Hinglish scam data that no public dataset gives us.
    HOW:  we alternate scam/benign so the synthetic half stays balanced, and we skip
          any message we have already produced. The skip matters: templates plus a
          finite list of placeholder values WILL collide, and duplicate text across
          the train/test boundary is leakage.
    HONEST NOTE: these are synthetic. Because templates repeat wording, the evaluation
          split holds out WHOLE FAMILIES - see make_leaksafe_split.
    """
    scam = [(f, t, 1) for f, t in SCAM_TEMPLATES.items()]
    benign = [(f, t, 0) for f, t in BENIGN_TEMPLATES.items()]
    rows, seen, attempts = [], set(), 0
    while len(rows) < n_messages and attempts < n_messages * 40:
        attempts += 1
        group = scam if len(rows) % 2 == 0 else benign      # alternate -> 50/50 labels
        family, templates, label = group[rng.integers(len(group))]
        text = fill_placeholders(str(rng.choice(templates)), rng)
        text = apply_spelling_variants(text, rng)
        if text in seen:                                    # already generated: try again
            continue
        seen.add(text)
        rows.append({"text": text, "label": label, "language_mix": "hinglish",
                     "source": "synthetic_hinglish", "template_family": family})
    print(f"  generated {len(rows)} unique synthetic messages in {attempts} attempts")
    return pd.DataFrame(rows)


def build_rings(messages):
    """
    Give every template family its own dedicated "rings" (campaigns).

    WHAT: returns a dict like {"kyc_expiry": ["kyc_expiry__r0", "kyc_expiry__r1"]}.
    WHY:  THIS IS WHAT MAKES THE SPLIT HONEST. One ring runs exactly one campaign, so
          a ring never mixes two kinds of wording. Because of that, holding out a ring
          and holding out a wording become the same action, and we do not need two
          competing split rules that fight each other.
          A real scam ring behaves this way too: it runs one script, not nine.
    HOW:  a big family gets more rings so we can hold out part of it; every family gets
          at least 2 so no family is all-or-nothing.
    """
    rings = {}
    for family, count in messages["template_family"].value_counts().items():
        n_rings = max(2, int(round(count / MESSAGES_PER_RING)))
        rings[family] = [f"{family}__r{k}" for k in range(n_rings)]
    return rings


def simulate_senders(messages, rng):
    """
    Invent the senders: assign them to rings and give each one a history.

    WHAT: returns one row per sender (id, ring, family, behaviour, reports, age).
    WHY:  F1 needs SenderTrust, F3 needs the graph shape. Real data has neither.
    THE KEY IDEA - `behaves_scammy` is NOT the same as `is_scammer`. We flip it for
    12% of scammers (they lie low) and 5% of innocent senders (they look odd), so no
    graph feature can simply be read off as the answer. Reports and account age follow
    BEHAVIOUR, not the hidden truth, which is what a real abuse team would observe.
    """
    family_label = messages.groupby("template_family")["label"].first()
    rings = build_rings(messages)
    rows = []
    for family, count in messages["template_family"].value_counts().items():
        is_scam_family = bool(family_label[family] == 1)
        # busy scam senders vs quiet normal senders -> the fan-out difference F3 needs
        per_sender = MESSAGES_PER_SCAM_SENDER if is_scam_family else MESSAGES_PER_BENIGN_SENDER
        n_senders = max(2, int(round(count / per_sender)))
        for index in range(n_senders):
            rows.append({
                "sender_id": f"S{len(rows):05d}",
                "sender_ring": rings[family][index % len(rings[family])],
                "template_family": family,
                "is_scammer": is_scam_family,
            })
    return _add_sender_history(pd.DataFrame(rows), rng)


def _choose_fraction(mask, fraction, rng):
    """
    Pick exactly `fraction` of the senders flagged by `mask`.

    WHAT: returns a boolean array that is True for the chosen senders only.
    WHY:  lets us say "exactly 12% of scammers hide" and have it be literally true,
          instead of 12% on average over imaginary repeated runs.
    EXAMPLE: _choose_fraction(is_scammer, 0.12, rng) -> 12% of the scammers are True
    """
    chosen = np.zeros(len(mask), dtype=bool)
    eligible = np.flatnonzero(mask)                  # positions where mask is True
    how_many = int(round(len(eligible) * fraction))
    if how_many > 0:
        picked = rng.choice(eligible, size=how_many, replace=False)
        chosen[picked] = True
    return chosen


def _add_sender_history(senders, rng):
    """
    Add the noisy behaviour flag plus the complaint/age history to each sender.

    WHAT: adds behaves_scammy, past_reports, account_age_days.
    WHY:  split out of simulate_senders only to keep each function short and testable.
    """
    n = len(senders)
    is_scammer = senders["is_scammer"].to_numpy()
    # Pick an EXACT number of each kind rather than rolling a dice per sender. With only
    # ~80 scam senders, a per-sender 12% dice can easily come out as 7%, and then the
    # 12% we document in the paper would not be the 12% in the data.
    hides = _choose_fraction(is_scammer, FRAC_SCAM_SENDERS_LOOK_NORMAL, rng)
    looks_odd = _choose_fraction(~is_scammer, FRAC_BENIGN_SENDERS_LOOK_ODD, rng)
    senders["behaves_scammy"] = (is_scammer & ~hides) | looks_odd
    scammy = senders["behaves_scammy"].to_numpy()
    # A fresh account with several complaints, versus an old account with almost none.
    #
    # THE RANGES DELIBERATELY OVERLAP. An earlier version used 1-120 days for scammy
    # accounts and 200-3000 for the rest - no overlap at all, so account age alone
    # almost gave the label away and the model leaned on it above everything else.
    # Overlapping them means a brand-new innocent account can look suspicious and a
    # patient scammer who aged their account can look fine, which is the real problem.
    senders["past_reports"] = np.where(scammy, rng.poisson(3, n), rng.poisson(0.6, n))
    senders["account_age_days"] = np.where(scammy, rng.integers(1, 400, n),
                                           rng.integers(30, 3000, n))
    senders["activity_weight"] = _activity_weights(scammy, rng)
    return senders


def _activity_weights(scammy, rng):
    """
    Decide how busy each account is.

    WHAT: returns a lognormal weight per sender; bigger means more messages.
    WHY:  drawn from a lognormal so the result is heavy tailed, like real traffic: most
          senders are quiet and a few send a great deal. Scammy behaviour shifts the
          average up 3x but does NOT separate the groups, because the tails overlap -
          which is exactly the point. With a fixed quota instead, message volume alone
          reached F1 = 1.00 and gave the whole answer away.
    """
    base = np.where(scammy, 3.0, 1.0)
    return rng.lognormal(np.log(base), ACTIVITY_SPREAD, len(scammy))


def _sender_number(sender_id):
    """
    Turn "S00042" into the integer 42.

    WHAT: a stable numeric id for a sender.
    WHY:  we need a repeatable way to give each sender its own slice of recipients and
          its own burst day. Python's built-in hash() CANNOT be used for this: hashing
          a string is randomised per process, so the "same" dataset would come out
          different on every run and RANDOM_SEED would be a lie.
    """
    return int(sender_id[1:])


def assign_senders(messages, senders, rng):
    """
    Give every message a sender, a recipient and a timestamp.

    WHAT: adds sender_id, recipient_id, sender_ring, timestamp columns.
    WHY:  this is what turns a pile of text into a network F3 can compute on.
    HOW:  a message is sent by one of the senders that runs its own campaign. Scammy
          senders write to a brand-new stranger every single time (unique-recipient
          ratio near 1.0); normal senders cycle round their few saved contacts
          (ratio well below 1.0). That contrast is the graph signal.
    """
    by_family = {f: g["sender_id"].tolist() for f, g in senders.groupby("template_family")}
    # probability of picking each sender, proportional to how active that account is
    weights = {f: (g["activity_weight"] / g["activity_weight"].sum()).to_numpy()
               for f, g in senders.groupby("template_family")}
    contacts = build_contact_lists(senders)
    lookup = senders.set_index("sender_id")
    sent_counts = {}                    # how many messages this sender has sent so far
    chosen, recipients, stamps = [], [], []
    for family, text_label in zip(messages["template_family"], messages["label"]):
        sender = str(rng.choice(by_family[family], p=weights[family]))
        position = sent_counts.get(sender, 0)
        sent_counts[sender] = position + 1
        scammy = bool(lookup.loc[sender, "behaves_scammy"])
        chosen.append(sender)
        recipients.append(_pick_recipient(contacts[sender], sender, position, scammy, rng))
        stamps.append(_pick_timestamp(sender, scammy, rng))
    out = messages.copy()
    out["sender_id"] = chosen
    out["recipient_id"] = recipients
    out["timestamp"] = pd.to_datetime(stamps)
    return out.merge(senders[["sender_id", "sender_ring"]], on="sender_id", how="left")


def build_contact_lists(senders):
    """
    Decide, for every sender, which people they message.

    WHAT: returns {sender_id: [recipient ids]}.
    WHY:  THIS IS WHAT GIVES THE GRAPH ITS SHAPE, and three of F3's features depend
          entirely on it:
            * Ordinary-looking accounts are placed in small social circles and mostly
              message each other. Because a recipient here is ANOTHER SENDER, messages
              flow both ways, friends-of-friends form triangles, and `clustering` and
              `pagerank` become meaningful. In a pure sender->recipient graph both are
              dead: there are no triangles and senders have no incoming edges at all.
            * An account that BEHAVES like a campaign gets a long victim list shared
              with the rest of its ring, so the ring shows up as a group.
            * Victims are drawn mostly from real people, because that is who gets
              scammed - not from a separate pool of inert ids.
    NOTE: everything here keys on `behaves_scammy`, never on `is_scammer`. The network
          can only see behaviour. So the 12% of scammers who lie low keep an innocent
          looking social circle, and the 5% of innocent senders who look odd get a
          spray list. Keying on the hidden truth would make the graph a perfect lie
          detector and every score in the paper meaningless.
    """
    ordinary = senders.loc[~senders["behaves_scammy"], "sender_id"].tolist()
    strangers = [f"R{index:05d}" for index in range(N_RECIPIENTS)]
    contacts = {}
    for position, sender in enumerate(senders["sender_id"]):
        contacts[sender] = _circle_of(position, sender, ordinary)
    for ring, group in senders[senders["behaves_scammy"]].groupby("sender_ring"):
        pool = _victim_pool(ring, ordinary, strangers)
        for sender in group["sender_id"]:
            contacts[sender] = pool          # every member of the ring shares the list
    return contacts


def _circle_of(position, sender, ordinary):
    """
    Find the few friends an ordinary-looking account messages.

    WHAT: returns up to BENIGN_CONTACTS other accounts from the same social circle.
    WHY:  keeping the circle small and CLOSED is what creates triangles in the graph,
          which is the only reason the clustering coefficient carries any information.
    """
    if not ordinary:
        return ["R00000"]
    first = (position // SOCIAL_CIRCLE * SOCIAL_CIRCLE) % len(ordinary)
    members = [p for p in ordinary[first:first + SOCIAL_CIRCLE] if p != sender]
    return members[:BENIGN_CONTACTS] or [ordinary[0]]


def _victim_pool(ring, people, strangers):
    """
    Build the shared victim list for one scam ring.

    WHAT: returns RING_VICTIM_POOL recipient ids, mostly real people.
    WHY:  a ring works from one list. Two accounts hitting the same victims is the
          clearest sign they are the same operation.
    HOW:  the starting point is derived from the ring NAME, so it is the same on every
          run (the same reason we never use Python's hash()).
    """
    start = sum(ord(character) for character in ring) * 17
    pool = []
    for step in range(RING_VICTIM_POOL):
        if step % 4 == 3:                              # one in four is a stranger
            pool.append(strangers[(start + step) % len(strangers)])
        else:
            pool.append(people[(start + step) % len(people)])
    return pool


def _pick_recipient(candidates, sender, position, scammy, rng):
    """
    Choose who this message goes to, from the sender's own contact list.

    WHAT: returns one recipient id.
    WHY:  scammy senders work steadily down a long shared list, so their recipients
          keep being new AND overlap with their ring partners. Normal senders cycle
          round their few friends.
    THE JITTER MATTERS: without it every scammy sender would reach a brand-new person
    every single time, so "unique recipients / messages sent == 1.0" would identify a
    scammer perfectly and F3 would get a free answer instead of a real problem.
    """
    if scammy:
        start = _sender_number(sender) * 7             # each member enters the list
        if position > 0 and rng.random() < 0.25:       # 25%: pester a victim again
            step = int(rng.integers(0, position))
        else:
            step = position
        return candidates[(start + step) % len(candidates)]
    if rng.random() < 0.15:                            # 15%: a normal person meets someone new
        return f"R{(_sender_number(sender) * 37 + position) % N_RECIPIENTS:05d}"
    return candidates[position % len(candidates)]


def _pick_timestamp(sender, scammy, rng):
    """
    Pick when a message was sent: scammy senders fire in bursts, others do not.

    WHAT: returns a pandas Timestamp inside a 60-day window.
    WHY:  "burstiness" is one of F3's graph features. A scam campaign blasts its whole
          list in a few hours; a person texts a few times a day for weeks.
    """
    base = pd.Timestamp("2026-01-01")
    if scammy:
        burst_day = _sender_number(sender) % 50       # stable, not process-dependent
        # each campaign has its OWN pace: some finish in 2 hours, some take 10 days
        window = BURST_HOURS_MIN + (_sender_number(sender) * 7) % (
            BURST_HOURS_MAX - BURST_HOURS_MIN)
        return base + pd.Timedelta(days=burst_day, minutes=int(rng.integers(0, window * 60)))
    return base + pd.Timedelta(days=int(rng.integers(0, 60)),
                               minutes=int(rng.integers(0, 24 * 60)))


def _hold_out_quarter(names, rng):
    """
    Pick a quarter of the given group names to become the test set.

    WHAT: returns a set of names (template families, or sender rings).
    WHY:  used for both halves of the leak-safe split. Returns an empty set when there
          is nothing to choose from, because a dataset built offline may have no real
          messages at all, and numpy raises if you sample from an empty list.
    """
    if len(names) == 0:
        return set()
    how_many = max(1, len(names) // 4)
    return set(rng.choice(names, size=how_many, replace=False))


def make_leaksafe_split(frame, rng):
    """
    Split train/test by CAMPAIGN, never by random row.

    WHAT: returns a Series of "train"/"test".
    WHY:  THIS IS THE MOST IMPORTANT METHODOLOGICAL CHOICE IN THE PROJECT.
          Our synthetic messages come from templates, so two rows of one family share
          most of their words. A random split would put near-identical twins in train
          and test, and the model would score ~99% by recognising wording it had
          already read. So:
            * SYNTHETIC families are held out WHOLE (every ring of that family goes to
              test). The model must handle a scam type it has never seen.
            * UCI / seed families are REAL text with no shared template, so holding out
              whole rings of them is enough; splitting a real family across train and
              test leaks nothing, because no two of those messages share wording.
          Because each ring runs exactly one campaign (see build_rings), a sender never
          appears on both sides either.
    """
    synthetic = frame["source"] == "synthetic_hinglish"
    families = sorted(frame.loc[synthetic, "template_family"].unique())
    real_rings = sorted(frame.loc[~synthetic, "sender_ring"].unique())
    # _hold_out_quarter guards against an empty list: a dataset might be synthetic-only
    # (built offline with no UCI download), and sampling from nothing raises.
    held_families = _hold_out_quarter(families, rng)
    held_rings = _hold_out_quarter(real_rings, rng)
    is_test = (synthetic & frame["template_family"].isin(held_families))
    is_test = is_test | ((~synthetic) & frame["sender_ring"].isin(held_rings))
    print(f"  leak-safe split: held out synthetic families "
          f"{sorted(str(f) for f in held_families)}")
    return pd.Series(np.where(is_test, "test", "train"), index=frame.index)


def make_time_split(frame, test_fraction=0.2):
    """
    Split train/test by TIME: the last slice of the window is the test set.

    WHAT: returns a Series of "train"/"test".
    WHY:  this is how the detector would really be used - trained on the past,
          run on messages that have not arrived yet. We report both splits because
          each one hides a different weakness.
    """
    cutoff = frame["timestamp"].quantile(1 - test_fraction)
    print(f"  time split cutoff: {cutoff}")
    return pd.Series(np.where(frame["timestamp"] > cutoff, "test", "train"), index=frame.index)


def build_dataset():
    """
    Run the whole pipeline and write both CSVs. This is the entry function.

    WHAT: returns the combined DataFrame and saves it to data/processed/.
    WHY:  one command rebuilds every input the other 8 functionalities need.
    EXAMPLE: python -m src.data_builder
    """
    rng = np.random.default_rng(RANDOM_SEED)     # one seed -> same dataset every time
    print("1. loading real English messages (UCI)...")
    uci = load_uci_sms()
    print("2. loading real Hinglish seed (Hugging Face), deduplicated against UCI...")
    seed = load_hinglish_seed(set(uci["text"].str.strip()))
    print(f"3. generating {N_SYNTHETIC_MESSAGES} synthetic Hinglish messages...")
    synthetic = generate_synthetic_messages(N_SYNTHETIC_MESSAGES, rng)
    messages = pd.concat([uci, seed, synthetic], ignore_index=True)
    messages = messages.drop_duplicates(subset="text").reset_index(drop=True)
    print("4. simulating the senders and the message network...")
    senders = simulate_senders(messages, rng)
    messages = assign_senders(messages, senders, rng)
    print(f"   {len(senders)} senders across {senders.sender_ring.nunique()} rings")
    messages["message_id"] = [f"M{i:06d}" for i in range(len(messages))]
    print("5. building the two evaluation splits...")
    messages["split_leaksafe"] = make_leaksafe_split(messages, rng)
    messages["split_time"] = make_time_split(messages)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    columns = ["message_id", "text", "sender_id", "recipient_id", "timestamp", "label",
               "language_mix", "source", "template_family", "sender_ring",
               "split_leaksafe", "split_time"]
    messages[columns].to_csv(COMBINED_CSV, index=False)
    senders.to_csv(PROCESSED_DIR / "senders.csv", index=False)
    print(f"6. wrote {len(messages)} messages -> {COMBINED_CSV}")
    return messages[columns]


# Run: python -m src.data_builder
if __name__ == "__main__":
    data = build_dataset()
    print("\n" + "=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)
    print(f"total messages : {len(data)}")
    print(f"scam / benign  : {int(data.label.sum())} / {int((1 - data.label).sum())}")
    print("\nby source:")
    print(data.groupby("source")["label"].agg(["count", "sum"]).to_string())
    print("\nleak-safe split:")
    print(pd.crosstab(data.split_leaksafe, data.label).to_string())
    print("\ntime split:")
    print(pd.crosstab(data.split_time, data.label).to_string())
    print("\nexample synthetic Hinglish scam message:")
    hinglish_scam = data[(data.label == 1) & (data.source == "synthetic_hinglish")]
    print(" ", hinglish_scam.iloc[0]["text"][:150])
    print("\nexample synthetic Hinglish benign message:")
    hinglish_benign = data[(data.label == 0) & (data.source == "synthetic_hinglish")]
    print(" ", hinglish_benign.iloc[0]["text"][:150])
