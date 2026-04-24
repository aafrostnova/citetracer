# Final Invalid Reasons

最后更新：2026-04-22

说明：以下 31 条是当前最终确认的 `invalid` 条目。判定标准不是“文献对象是否曾经存在”，而是“当前 bib 中给出的精确 URL 是否还能访问到与该 bib 条目相匹配的对象”。因此，有些条目对应的对象是真实存在的，但由于 URL 已失效、跳转到别处、只剩首页/组织页、或页面内容已变化，仍被保留为 `invalid`。

1. `sternberg1999lectures`
   - bib 写的是 1999 年 Shlomo Sternberg 的《Lectures on differential geometry》。
   - 但 URL 指向的是 `ams.org` 上一份 1965 年 AMS Bulletin 相关 PDF，不是这本 1999 年的书。
   - 所以属于“URL 指向了错误对象”。

2. `eval-harness`
   - bib 标题是 `A framework for few-shot language model evaluation`。
   - 当前 Zenodo 记录 `10.5281/zenodo.10256836` 的标题是 `EleutherAI/lm-evaluation-harness: Major refactor`，对应的是软件发布记录，而不是 bib 中写的这个标题。
   - DOI 和版本能对上，但页面标题对象不对，所以保留为 `invalid`。

3. `anthropic2024claude3`
   - bib 指向 Papers with Code 上的 Claude 3 论文页面。
   - 当前该 URL 会跳到 Hugging Face 的 `Trending Papers` 页面，不再是 Claude 3 的具体论文页。
   - 这是“重定向到了不相关页面”。

4. `skyworkcritic2024`
   - bib 写的是 `Skywork Critic Model Series`。
   - 现在 URL 只是 Hugging Face 上 `Skywork` 组织主页，不是具体的 `Critic Model Series` 页面。
   - 组织页不足以支撑这个具体条目。

5. `block_imaging_inc_2024_2022`
   - bib 标题是 `2024 MRI Machine Price Guide`。
   - 当前页面标题已经变成 `How Much Does an MRI Machine Cost in 2026?`。
   - 页面内容明显更新成别的年份版本，已不再匹配原条目。

6. `cylanceprotect`
   - bib 指向的是 `CylancePROTECT Malware Execution Control` 的 PDF。
   - 当前 URL 会跳到 `arcticwolf.com/cylance/` 的落地页，不再是原 PDF。
   - 这是“原始 PDF 链接失效并跳转到品牌/产品营销页”。

7. `havaei2025jailbreak`
   - bib 标题是 `The Jailbreak Cookbook`。
   - 当前 URL 页面标题是 `Post Not Found | General Analysis`，正文也显示文章不存在。
   - 对应文章已经不可用，所以是 `invalid`。

8. `CROSSBOW`
   - bib 表示的是 `XBOW Sensor Motes Specifications`。
   - 现在 `xbow.com` 打开的是 `Autonomous Offensive Security Platform`，是完全不同的公司主页内容。
   - 同域名已对应别的对象。

9. `TUGInstmem`
   - bib URL 写成了 `http://wwtug.org/instmem.html`。
   - 这个 host 现在不解析；能找到相近内容的是 `tug.org/instmem.html`，不是 bib 里给出的精确 URL。
   - 所以当前给定 URL 本身无效。

10. `juravsky2025tokasaurus`
   - bib 标题是 `Tokasaurus: An LLM Inference Engine for High-Throughput Workloads`。
   - URL 却指向 `fla-org/flash-linear-attention` GitHub 仓库。
   - 这是明显“条目标题和 URL 对象完全不一致”。

11. `gemini25pro`
   - bib 写的是 `Gemini 2.5 Pro` 页面。
   - 当前这个 URL 最终跳到的是 `Gemini 3.1 Pro — Google DeepMind`。
   - 版本对象已经变了，不再是同一个条目。

12. `mistral_large_2024`
   - bib 标题是 `Mistral Large: A General-Purpose Language Model`。
   - 当前页面是 `Large Enough`，正文明确在讲 `Mistral Large 2`。
   - 这不是 bib 中引用的那个页面对象。

13. `AUVSI16`
   - bib 标题是 `UAS Aid in South Carolina Tornado Investigation`。
   - 虽然外部二次来源仍提到这个 URL，但当前精确 URL 直接返回 `404`。
   - 由于现行 URL 无法访问原对象，按当前可访问性标准仍是 `invalid`。

14. `liu2025there`
   - bib 指向一个 Notion 页面，标题是 `There May Not be Aha Moment in R1-Zero-like Training — A Pilot Study`。
   - 现在打开后只是通用 `Notion` 页面，看不到该具体文章内容。
   - URL 无法支撑该具体条目。

15. `song2025seed-full`
   - bib 条目是 `Seed Diffusion` 的具体论文/技术报告页面。
   - 当前 URL 只落到通用的 ByteDance Seed 页面，不是这篇具体条目页。
   - 这是“首页/品牌页替代了具体对象页”。

16. `doubao-seed`
   - bib 指向 `Doubao-Seed-1.6`。
   - 当前页面标题只有 `字节跳动Seed`，没有 `Doubao-Seed-1.6` 的具体信息。
   - 仍然只是泛化主页，不足以匹配该条目。

17. `escher`
   - bib 写的是 M.C. Escher 的具体作品 `Cycles`。
   - URL 现在只是 Escher 的作品总画廊首页，没有明确到 `Cycles` 这幅作品。
   - 画廊总页不能等同于该具体作品页面。

18. `google2025gemini25pro`
   - bib 标题是 `Gemini 2.5 Pro`。
   - 当前 `https://deepmind.google/models/gemini/pro/` 页面标题是 `Gemini 3.1 Pro — Google DeepMind`。
   - URL 已经服务于更新后的版本对象，所以原条目变成了假阳性并被打回 `invalid`。

19. `nanogpt_issue303`
   - bib 说的是 `Issue #303: Gradient explosion when training with bfloat16`。
   - 当前 Issue #303 的真实标题是 `Train/Val Loss Issues when training GPT-2 from OWT`。
   - Issue 编号虽然对上，但标题主题不对，说明 bib 里的描述和 URL 对象不匹配。

20. `Lisa_My_Research_Software_2017`
   - bib 标题是 `My Research Software`。
   - URL 实际指向的是 GitHub 上的 `github-linguist/linguist` 仓库。
   - 这是完全不相关的项目。

21. `text-davinci-003`
   - bib 目标是具体模型 `text-davinci-003`。
   - URL 却只是 `https://openai.com` 首页。
   - 首页无法证明这个具体模型条目，且并非模型详情页。

22. `Sanseviero2024LLM`
   - bib 标题是 `LLM Evals and Benchmarking`。
   - 当前 URL 是 Omar Sanseviero 的个人主页 `hackerllama`，不是这篇具体文章。
   - 个人主页不能替代具体博文/页面。

23. `maa2024aime24`
   - bib 指向 2024 年 AIME 页面。
   - 当前 URL 会跳到 `2024-25 AIME Thresholds Are Available` 新闻页。
   - 这不再是 2024 AIME 的目标页面。

24. `maa2025aime25`
   - 与上一条同理。
   - 当前相同 URL 仍然只指向阈值公告页，而不是 2025 AIME 的具体页面。
   - 所以继续判为 `invalid`。

25. `skywork-o1`
   - bib 条目是 `Skywork-o1 Open Series`。
   - URL 只是 Hugging Face 上 `Skywork` 的组织主页。
   - 组织主页不足以表示具体模型系列。

26. `simplerl`
   - bib 标题是具体研究文章：`7B Model and 8K Examples: Emerging Reasoning with Reinforcement Learning is Both Effective and Efficient`。
   - 当前 URL 打开只是通用 `Notion` 页，没有可确认的具体文章内容。
   - 无法证明该精确条目。

27. `needle`
   - bib URL 写成了 `https://github.com/gkamradt/LLMTest NeedleInAHaystack.`，中间有空格。
   - 这个 URL 本身就是 malformed，无法正确定位到仓库。
   - 虽然真实项目存在，但 bib 里给出的 URL 无效。

28. `gpt4.1`
   - bib 标题写的是 `GPT-4 Technical Report`。
   - 当前 URL 页面标题是 `Introducing GPT-4.1 in the API`。
   - 这是 GPT-4.1 公告页，不是所引的技术报告页。

29. `midjourneyv7`
   - bib 标题是 `midjourney v7`。
   - URL 实际只是 `https://github.com/midjourney` 组织主页。
   - 它不是 v7 的发布说明，也不是对应模型页。

30. `url:nistchallenge`
   - bib 写的是 `Differential Privacy Synthetic Data Challenge`。
   - 该挑战曾经存在，外部资料也能找到引用，但当前精确 Challenge.gov URL 已经返回 `404`。
   - 由于这里检查的是“当前给定 URL 是否仍有效”，所以仍保留 `invalid`。

31. `tot`
   - bib 关联的是 `Tree of Thoughts` 的代码仓库。
   - bib 里写的是 `github.com/ysymyth/tree-of-thought- llm`，中间多了空格，而且当前实际 URL 也是 `404`。
   - 这是一个错误/损坏的仓库 URL，因此判为 `invalid`。

