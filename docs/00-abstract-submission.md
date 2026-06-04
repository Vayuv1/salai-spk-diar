### Title

Enhancing Automatic ATC Information Extraction via Speaker Diarization

### Synopsis

Air traffic control communications convey safety-critical information within short time windows and often under noisy operational conditions. Pilots must interpret instructions from multiple speakers while managing other cockpit tasks, increasing cognitive workload and the risk of misinterpretation. Speech and language technologies offer the potential to automatically transcribe ATC communications and extract operationally relevant information to support pilot decision-making. However, frequent speaker alternation, overlapping transmissions, and variable audio quality continue to limit reliable real-time processing. This work investigates speaker turn detection as a key enabler for real-time ATC information extraction. Rather than relying on post-processed transcriptions, speaker diarization is integrated directly into ATC-focused automatic speech recognition pipelines. The study examines limitations of existing diarization methods and explores strategies to improve robustness. A small ATC-specific dataset with speaker annotations is constructed to support fine-tuning and evaluation. Results demonstrate improved speaker attribution and more reliable downstream information extraction, supporting safer ATC assistance.

### Extended abstract

Air traffic control (ATC) communications form a critical information channel in aviation operations, directly influencing pilot decision-making, situational awareness, and flight safety. These communications are often rapid, information-dense, and subject to challenging acoustic conditions, including background noise, radio interference, and overlapping transmissions. As traffic density and operational complexity increase, pilots must process large volumes of spoken instructions in real time, contributing to elevated cognitive workload and increased risk of misinterpretation.

Automatic speech recognition (ASR) and natural language processing (NLP) technologies have shown promise for transcribing ATC communications and extracting operationally relevant information, such as clearances, headings, altitudes, and frequency changes. Such capabilities could support pilot decision-making by enabling automatic population of avionics displays, real-time monitoring tools, and advisory systems. However, the reliability of ASR and NLP based ATC systems remains limited by several domain specific challenges, including informal or non-standard phraseology, speaker accents, variable transmission quality, and frequent alternation between controllers and multiple pilots.

A particularly critical challenge for real-time ATC information extraction is accurate speaker turn detection. Many existing ATC speech processing approaches rely on post-processed transcriptions, where speaker segmentation is performed after the entire audio stream has been processed. While effective for offline analysis, this paradigm is poorly suited to real-time applications, where timely attribution of utterances to the correct speaker is essential for downstream processing and decision support. Errors in speaker attribution can propagate through integrated systems, leading to incorrect interpretation of commands and reduced trust in automation.

This work focuses on integrating speaker diarization into ATC-focused ASR pipelines as a means of improving real-time information extraction. The study examines the behavior of state-of-the-art diarization approaches when applied to ATC communications, with particular attention to failure modes that are problematic in operational settings. One observed limitation is the tendency of some diarization systems to become locked into incorrect speaker assignments, especially in scenarios with short speaker turns or overlapping transmissions. Such errors are especially detrimental in safety-critical contexts, where misattributing a command to the wrong speaker can have serious consequences.

To address these challenges, multiple strategies are investigated to improve diarization robustness in the ATC domain. These include approaches for mitigating speaker assignment drift, techniques for handling rapid speaker alternation, and fine-tuning methods tailored to aviation-specific speech characteristics. Rather than treating diarization as an isolated task, the study emphasizes its role as an enabling component within a broader real-time ATC information extraction pipeline.

To support fine-tuning and evaluation, a small, application-specific ATC dataset with speaker annotations is manually constructed. The dataset is designed to reflect realistic ATC communication scenarios, including multiple speakers, short utterances, and variable audio quality. This dataset enables controlled evaluation of diarization performance and analysis of error patterns that directly impact downstream information extraction.

The experimental evaluation assesses diarization performance using standard metrics, complemented by analyses focused on operational relevance, such as speaker confusion during critical command exchanges. Preliminary results indicate that the proposed strategies can improve diarization robustness and lead to more reliable extraction of ATC information in real-time settings. These findings suggest that speaker diarization, when properly integrated and adapted to the ATC domain, can play a meaningful role in reducing pilot workload and supporting safer aviation operations.

This work contributes to ongoing efforts to qualify AI-based technologies for safety-critical avionics applications. By examining speaker diarization from a system-level perspective and grounding the evaluation in realistic ATC scenarios, the study aims to inform the design and deployment of trustworthy, real-time ATC assistance systems in the age of AI.

### Reviews

#### Review 1

**Points in favor**
Very important topic that could have significant impact.

**Points against**
Not sure about the specific systems and software that was/will be used.
How many samples, length of samples, noise levels, etc. are not mentioned.

**Recommended decision**
2 weak accept

**Suggested field of interest**
Communications, Navigation, and Surveillance and Information Networks

#### Review 2

**Points in favor**
The work addresses a high-stakes, real-world problem—pilot cognitive workload and the risk of miscommunication. By focusing on the reliability of ATC information extraction, the research has a direct and clear path toward enhancing aviation safety.
A major strength is the shift away from post-processed, offline transcriptions. By integrating speaker diarization directly into the live ASR pipeline, the authors address the latency requirements essential for cockpit decision-support tools.

**Points against**
N/A

**Recommended decision**
3 accept

**Suggested field of interest**
Communications, Navigation, and Surveillance and Information Networks

#### Review 3

**Points in favor**
Directly targets safety-critical ATC domain
Uses real ATC-specific dataset
System-level integration view is provided
Evaluation plan is convincing

**Points against**
Dataset size appears small
Regulatory or certification plans not mentioned

**Recommended decision**
3 accept

**Suggested field of interest**
Avionics Platforms