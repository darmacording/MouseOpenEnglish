import express from "express";
import path from "path";
import { fileURLToPath } from "url";
import { GoogleGenAI, Type, Modality } from "@google/genai";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = 3000;

// Support high payload size for high-res photo uploads
app.use(express.json({ limit: "50mb" }));
app.use(express.urlencoded({ extended: true, limit: "50mb" }));

// Helper to resiliently parse and repair JSON returned by Gemini AI models
function safeJsonParse(rawText, defaultValue = null) {
  if (!rawText || typeof rawText !== "string") {
    if (defaultValue !== null) return defaultValue;
    throw new Error("AI 응답 데이터가 비어 있습니다.");
  }

  let cleaned = rawText.trim();
  // Strip markdown code fences
  cleaned = cleaned.replace(/^```(?:json)?\s*/i, "").replace(/\s*```$/i, "").trim();

  // 1. Standard JSON.parse
  try {
    const data = JSON.parse(cleaned);
    if (data && typeof data === "object") return data;
  } catch (initialErr) {
    // 2. Clean unescaped newlines/tabs inside quotes & trailing commas
    try {
      let sanitized = cleaned.replace(/"(?:[^"\\]|\\.)*"/gs, (str) => {
        return str.replace(/[\n\r]/g, "\\n").replace(/\t/g, "\\t");
      });
      sanitized = sanitized
        .replace(/,\s*([}\]])/g, "$1")
        .replace(/[\u0000-\u0008\u000B-\u000C\u000E-\u001F]+/g, "");
      const data = JSON.parse(sanitized);
      if (data && typeof data === "object") return data;
    } catch (e2) {}

    // 3. Extract between first { and last }
    const firstBrace = cleaned.indexOf("{");
    const lastBrace = cleaned.lastIndexOf("}");
    if (firstBrace !== -1 && lastBrace !== -1 && lastBrace > firstBrace) {
      const candidate = cleaned.substring(firstBrace, lastBrace + 1);
      try {
        const data = JSON.parse(candidate);
        if (data && typeof data === "object") return data;
      } catch (e3) {
        try {
          let sanitized = candidate.replace(/"(?:[^"\\]|\\.)*"/gs, (str) => {
            return str.replace(/[\n\r]/g, "\\n").replace(/\t/g, "\\t");
          });
          sanitized = sanitized
            .replace(/,\s*([}\]])/g, "$1")
            .replace(/[\u0000-\u0008\u000B-\u000C\u000E-\u001F]+/g, "");
          const data = JSON.parse(sanitized);
          if (data && typeof data === "object") return data;
        } catch (e4) {}
      }
    }

    // 4. Regex-based field recovery fallback
    const result = {
      extractedText: "",
      pureEnglishText: "",
      passageSummary: "오늘의 EBS 입트영 학습 내용",
      passageSentences: [],
      vocabList: []
    };

    const extMatch = cleaned.match(/"extractedText"\s*:\s*"((?:[^"\\]|\\.)*)"/s);
    if (extMatch) result.extractedText = extMatch[1].replace(/\\n/g, "\n").replace(/\\"/g, "\"").replace(/\\\\/g, "\\");

    const pureMatch = cleaned.match(/"pureEnglishText"\s*:\s*"((?:[^"\\]|\\.)*)"/s);
    if (pureMatch) result.pureEnglishText = pureMatch[1].replace(/\\n/g, "\n").replace(/\\"/g, "\"").replace(/\\\\/g, "\\");

    const sumMatch = cleaned.match(/"passageSummary"\s*:\s*"((?:[^"\\]|\\.)*)"/s);
    if (sumMatch) result.passageSummary = sumMatch[1].replace(/\\n/g, "\n").replace(/\\"/g, "\"");

    // Extract vocabList items via regex
    const vocabItemRegex = /\{\s*"id"\s*:\s*(\d+)[^}]*?"english"\s*:\s*"([^"]+)"[^}]*?"korean"\s*:\s*"([^"]+)"/g;
    let vMatch;
    while ((vMatch = vocabItemRegex.exec(cleaned)) !== null) {
      result.vocabList.push({
        id: parseInt(vMatch[1], 10),
        type: "word",
        english: vMatch[2],
        korean: vMatch[3],
        phonetic: "",
        exampleEn: "",
        exampleKo: ""
      });
    }

    if (result.extractedText || result.pureEnglishText || result.vocabList.length > 0) {
      return result;
    }

    if (defaultValue !== null) return defaultValue;
    throw new Error(`AI 응답 형식 처리 중 오류 (${initialErr.message})`);
  }
}

// Unified Gemini Generator with automatic model and API key fallback
async function generateWithFallback({
  userApiKey,
  prompt,
  parts,
  contents,
  systemInstruction,
  responseSchema,
  responseMimeType,
  speechConfig,
  responseModalities,
  preferredModel = "gemini-3.1-flash-lite",
  maxOutputTokens = 8192,
  timeoutMs = 25000
}) {
  const modelsToTry = [
    preferredModel,
    "gemini-3.1-flash-lite",
    "gemini-3.6-flash"
  ].filter(Boolean).filter((v, i, a) => a.indexOf(v) === i);

  const keysToTry = [];
  if (userApiKey && typeof userApiKey === "string" && userApiKey.trim().length > 10) {
    keysToTry.push({ key: userApiKey.trim(), isUserKey: true });
  }
  if (process.env.GEMINI_API_KEY && !keysToTry.some(k => k.key === process.env.GEMINI_API_KEY)) {
    keysToTry.push({ key: process.env.GEMINI_API_KEY, isUserKey: false });
  }

  if (keysToTry.length === 0) {
    throw new Error("GEMINI_API_KEY 환경변수 또는 사용자 API 키가 설정되지 않았습니다.");
  }

  let lastError = null;

  for (const { key: apiKey, isUserKey } of keysToTry) {
    const ai = new GoogleGenAI({
      apiKey,
      httpOptions: {
        headers: {
          "User-Agent": "aistudio-build",
        },
      },
    });

    for (const model of modelsToTry) {
      try {
        const config = {
          maxOutputTokens,
        };
        if (systemInstruction) config.systemInstruction = systemInstruction;
        if (responseMimeType) config.responseMimeType = responseMimeType;
        if (responseSchema) config.responseSchema = responseSchema;
        if (speechConfig) config.speechConfig = speechConfig;
        if (responseModalities) config.responseModalities = responseModalities;

        let contentPayload;
        if (contents) {
          contentPayload = contents;
        } else if (parts) {
          contentPayload = [{ role: "user", parts }];
        } else if (prompt) {
          contentPayload = prompt;
        }

        // Fast timeout wrapper per model attempt
        const timeoutPromise = new Promise((_, reject) => {
          setTimeout(() => reject(new Error(`Model ${model} timeout after ${timeoutMs}ms`)), timeoutMs);
        });

        const response = await Promise.race([
          ai.models.generateContent({
            model,
            contents: contentPayload,
            config,
          }),
          timeoutPromise
        ]);

        return response;
      } catch (err) {
        lastError = err;
        console.warn(`[Gemini Fallback] Model ${model} failed with ${isUserKey ? 'User Key' : 'Server Key'}: ${err.message?.substring(0, 120)}`);
        
        // If user's custom key failed for any reason, skip remaining models on this key and go to server key immediately
        if (isUserKey) {
          console.warn("[Gemini Fallback] User key failed, falling back to server key...");
          break;
        }
      }
    }
  }

  throw lastError || new Error("AI 모델 호출에 실패했습니다. 잠시 후 다시 시도해 주세요.");
}

// Health & configuration check
app.get("/api/config", (req, res) => {
  res.json({
    hasServerKey: !!process.env.GEMINI_API_KEY,
    status: "ok",
  });
});

// OCR & 10 Vocabulary Extraction
app.post("/api/analyze", async (req, res) => {
  try {
    const { imageBase64, text, userApiKey } = req.body;
    if (!imageBase64 && (!text || !text.trim())) {
      return res.status(400).json({ error: "본문 사진이나 텍스트를 제공해 주세요." });
    }

    const systemInstruction = `You are an expert OCR and English learning engine for EBS '입이 트이는 영어' (입트영).

YOUR MISSION:
1. STRICT OCR: Transcribe all English and Korean text visible in the user's photo/input accurately.
2. Return a SINGLE valid JSON object with the following schema:
{
  "extractedText": "Complete transcribed textbook passage text with Korean notes/translations",
  "pureEnglishText": "ONLY the clean English passage sentences without headers or Korean, ready for audio narration",
  "passageSummary": "한 줄 요약 (예: 재택근무 트렌드와 원활한 협업 노하우)",
  "passageSentences": [
    {
      "id": 1,
      "english": "Exact English sentence 1",
      "korean": "자연스러운 한국어 번역 1",
      "targetKeyword": "중심 표현 또는 빈칸"
    }
  ],
  "vocabList": [
    {
      "id": 1,
      "type": "word" or "expression",
      "english": "key word or phrase",
      "korean": "핵심 의미",
      "phonetic": "/발음기호/",
      "exampleEn": "Natural example sentence containing this word",
      "exampleKo": "예문 한국어 해석"
    }
  ]
}
3. IMPORTANT: Extract EXACTLY 10 key items in "vocabList" (words and expressions) appearing in or closely related to the textbook passage.
4. Extract ALL sequential sentences from the passage in "passageSentences" for complete writing practice.`;

    const parts = [];
    if (imageBase64) {
      let base64Data = imageBase64;
      let mimeType = "image/jpeg";

      if (imageBase64.includes(",")) {
        const [header, data] = imageBase64.split(",");
        base64Data = data.trim();
        const mimeMatch = header.match(/data:([^;]+);/);
        if (mimeMatch) {
          mimeType = mimeMatch[1];
        }
      }

      parts.push({
        inlineData: {
          mimeType: mimeType || "image/jpeg",
          data: base64Data,
        },
      });
    }

    if (text && text.trim()) {
      parts.push({
        text: `Transcribe and analyze this EBS English passage:\n${text}\n\nReturn the complete JSON object with extractedText, pureEnglishText, passageSummary, all passageSentences, and exactly 10 vocabList items.`,
      });
    } else {
      parts.push({
        text: "Perform accurate OCR on this textbook photo. Extract the full passage text, clean English audio text, Korean summary, ALL sentences in order, and EXACTLY 10 key vocabulary/expression items as JSON.",
      });
    }

    const response = await generateWithFallback({
      userApiKey,
      parts,
      preferredModel: "gemini-3.1-flash-lite",
      systemInstruction,
      responseMimeType: "application/json",
    });

    const parsedData = safeJsonParse(response.text, {});

    // Ensure fallback defaults if model returned partial fields
    if (!parsedData.extractedText || typeof parsedData.extractedText !== "string") {
      parsedData.extractedText = parsedData.pureEnglishText || "EBS 입이 트이는 영어 본문";
    }
    if (!parsedData.pureEnglishText || typeof parsedData.pureEnglishText !== "string") {
      parsedData.pureEnglishText = parsedData.extractedText;
    }
    if (!parsedData.passageSummary || typeof parsedData.passageSummary !== "string") {
      parsedData.passageSummary = "EBS 입이 트이는 영어 오늘의 학습 본문";
    }
    if (!Array.isArray(parsedData.passageSentences) || parsedData.passageSentences.length === 0) {
      // Auto split sentences from English text
      const sList = (parsedData.pureEnglishText || parsedData.extractedText)
        .split(/(?<=[.?!])\s+/)
        .filter((s) => s.trim().length > 3);
      parsedData.passageSentences = sList.map((sent, idx) => ({
        id: idx + 1,
        english: sent.trim(),
        korean: "해당 문장을 우리말로 해석해보세요.",
        targetKeyword: "",
      }));
    }
    if (!Array.isArray(parsedData.vocabList)) {
      parsedData.vocabList = [];
    }

    // Ensure each vocabList item has valid properties
    parsedData.vocabList = parsedData.vocabList.map((item, idx) => ({
      id: item.id || idx + 1,
      type: item.type === "word" ? "word" : "expression",
      english: item.english || `Expression ${idx + 1}`,
      korean: item.korean || "핵심 표현",
      phonetic: item.phonetic || "",
      exampleEn: item.exampleEn || item.english || "",
      exampleKo: item.exampleKo || item.korean || "",
    }));

    res.json({ success: true, data: parsedData });
  } catch (error) {
    console.error("Analysis Error:", error);
    res.status(500).json({ error: error.message || "본문 분석 중 오류가 발생했습니다." });
  }
});

// Gemini TTS
app.post("/api/tts", async (req, res) => {
  try {
    const { text, userApiKey } = req.body;
    if (!text || !text.trim()) {
      return res.status(400).json({ error: "음성으로 변환할 텍스트가 없습니다." });
    }

    const promptText = `Please read the following EBS English passage clearly, naturally, and fluently with a warm native American radio broadcasting accent (EBS Radio Host style):\n\n${text}`;

    const keysToTry = [];
    if (userApiKey && typeof userApiKey === "string" && userApiKey.trim().length > 10) {
      keysToTry.push(userApiKey.trim());
    }
    if (process.env.GEMINI_API_KEY && !keysToTry.includes(process.env.GEMINI_API_KEY)) {
      keysToTry.push(process.env.GEMINI_API_KEY);
    }

    let lastError = null;
    let audioData = null;
    let mimeType = null;

    for (const apiKey of keysToTry) {
      try {
        const ai = new GoogleGenAI({
          apiKey,
          httpOptions: { headers: { "User-Agent": "aistudio-build" } },
        });

        const response = await ai.models.generateContent({
          model: "gemini-3.1-flash-tts-preview",
          contents: [{ parts: [{ text: promptText }] }],
          config: {
            responseModalities: [Modality.AUDIO],
            speechConfig: {
              voiceConfig: {
                prebuiltVoiceConfig: { voiceName: "Aoede" },
              },
            },
          },
        });

        const part = response.candidates?.[0]?.content?.parts?.[0];
        audioData = part?.inlineData?.data;
        mimeType = part?.inlineData?.mimeType;

        if (audioData && mimeType) {
          return res.json({ success: true, audioData, mimeType });
        }
      } catch (err) {
        lastError = err;
        console.warn(`[TTS Fallback] key len ${apiKey.length} failed: ${err.message}`);
      }
    }

    throw lastError || new Error("오디오 데이터를 생성하지 못했습니다.");
  } catch (error) {
    console.error("TTS Error:", error);
    res.status(500).json({ error: error.message || "TTS 음성 생성 중 오류가 발생했습니다." });
  }
});

// AI Nuance Explanation
app.post("/api/nuance", async (req, res) => {
  try {
    const { english, korean, userApiKey } = req.body;
    if (!english) {
      return res.status(400).json({ error: "단어/표현이 지정되지 않았습니다." });
    }

    const prompt = `Act as an EBS English teacher. Explain "${english}" (${korean || ""}) for Korean learners.
Return concise HTML (clean tags, bullet points):
1. 원어민 뉘앙스 차이 (Native Nuance)
2. 콩글리시 피하기 팁 (Konglish Caution)
3. 세련된 대체 표현 2개`;

    const response = await generateWithFallback({
      userApiKey,
      prompt,
      preferredModel: "gemini-3.1-flash-lite",
    });

    const text = response.text || "분석을 가져올 수 없습니다.";
    res.json({ success: true, html: text });
  } catch (error) {
    console.error("Nuance Error:", error);
    res.status(500).json({ error: error.message || "뉘앙스 분석 중 오류가 발생했습니다." });
  }
});

// AI Grounded News Example
app.post("/api/grounded-news", async (req, res) => {
  try {
    const { english, korean, userApiKey } = req.body;
    if (!english) {
      return res.status(400).json({ error: "단어/표현이 지정되지 않았습니다." });
    }

    const prompt = `You are a real-time global English media corpus curator for EBS English learners.
Target vocabulary/expression: "${english}" (Korean meaning: "${korean || ""}").

Provide TWO distinct, highly realistic and authentic recent headlines/sentences as used in major global media (such as BBC, The New York Times, CNN, Forbes, TechCrunch, or Wall Street Journal).

Format your response in clean, attractive HTML:
<div class="space-y-2">
  <div class="bg-white p-2.5 rounded-lg border border-emerald-200">
    <div class="flex items-center gap-1.5 font-bold text-emerald-900 text-[11px] mb-1">
      <span class="bg-emerald-600 text-white px-1.5 py-0.5 rounded text-[10px]">BBC / NYT Style 1</span>
      <span>[Headline / Media Context]</span>
    </div>
    <p class="text-slate-800 font-medium text-xs">"[Actual English Sentence containing '${english}']"</p>
    <p class="text-slate-500 text-[11px] mt-0.5">(자연스러운 한글 해석)</p>
  </div>
  <div class="bg-white p-2.5 rounded-lg border border-emerald-200">
    <div class="flex items-center gap-1.5 font-bold text-emerald-900 text-[11px] mb-1">
      <span class="bg-emerald-600 text-white px-1.5 py-0.5 rounded text-[10px]">Global Media Trend 2</span>
      <span>[Business / Life Context]</span>
    </div>
    <p class="text-slate-800 font-medium text-xs">"[Second English Sentence containing '${english}']"</p>
    <p class="text-slate-500 text-[11px] mt-0.5">(자연스러운 한글 해석)</p>
  </div>
</div>`;

    const response = await generateWithFallback({
      userApiKey,
      prompt,
      preferredModel: "gemini-3.1-flash-lite",
    });

    let html = response.text || "";
    html = html.replace(/```html/gi, "").replace(/```/g, "").trim();
    res.json({ success: true, html });
  } catch (error) {
    console.error("News Example Error:", error);
    res.status(500).json({ error: error.message || "글로벌 뉴스 예시 검색 중 오류가 발생했습니다." });
  }
});

// AI Writing Quiz Review & Feedback
app.post("/api/writing-review", async (req, res) => {
  try {
    const { koreanSentence, userEnglish, targetExpression, referenceEnglish, userApiKey } = req.body;
    if (!koreanSentence || !userEnglish) {
      return res.status(400).json({ error: "한글 문장과 작성한 영작 문장을 입력해 주세요." });
    }

    const systemInstruction = `You are a warm, supportive, expert EBS English tutor specializing in English sentence writing (영작문 첨삭 지도) for Korean learners.

Task:
Evaluate the student's English translation of the given Korean sentence.
- Korean Sentence: "${koreanSentence}"
- Student's English Translation: "${userEnglish}"
- Target Expression/Keyword (if any): "${targetExpression || "N/A"}"
- Reference Standard Translation: "${referenceEnglish || "N/A"}"

Evaluation Criteria:
1. Meaning Accuracy: Does it accurately convey the Korean meaning?
2. Grammar & Structure: Check articles (a/an/the), prepositions, verb tenses, subject-verb agreements, and collocations.
3. Natural Nuance: Is it natural native American English (EBS Radio broadcast style)?
4. Target Expression Usage: Did the user apply the target phrase naturally?

Provide structured JSON feedback with score (0-100), concise evaluation badge, polished correction of user's sentence, friendly Korean explanation of what went well and what to fix, 1-3 key bullet points, and 2 natural native variations.`;

    const response = await generateWithFallback({
      userApiKey,
      prompt: `Please evaluate this student's writing:
Korean: "${koreanSentence}"
Student English: "${userEnglish}"
Target Phrase: "${targetExpression || ""}"
Reference: "${referenceEnglish || ""}"`,
      preferredModel: "gemini-3.1-flash-lite",
      systemInstruction,
      responseMimeType: "application/json",
      responseSchema: {
        type: Type.OBJECT,
        properties: {
          score: { type: Type.INTEGER },
          evaluation: { type: Type.STRING },
          isAccurate: { type: Type.BOOLEAN },
          correctedEnglish: { type: Type.STRING },
          feedback: { type: Type.STRING },
          keyPoints: {
            type: Type.ARRAY,
            items: { type: Type.STRING },
          },
          nativeAlternatives: {
            type: Type.ARRAY,
            items: {
              type: Type.OBJECT,
              properties: {
                english: { type: Type.STRING },
                koreanNuance: { type: Type.STRING },
              },
              required: ["english", "koreanNuance"],
            },
          },
        },
        required: ["score", "evaluation", "isAccurate", "correctedEnglish", "feedback", "keyPoints", "nativeAlternatives"],
      },
    });

    const result = safeJsonParse(response.text);
    res.json({ success: true, review: result, data: result });
  } catch (error) {
    console.error("Writing Review Error:", error);
    res.status(500).json({ error: error.message || "영작 첨삭 중 오류가 발생했습니다." });
  }
});

// AI Chat Tutor
app.post("/api/chat", async (req, res) => {
  try {
    const { messages, targetWords, userApiKey } = req.body;
    if (!messages || !Array.isArray(messages)) {
      return res.status(400).json({ error: "메시지 내역이 올바르지 않습니다." });
    }

    const systemPrompt = `You are an encouraging native English tutor for Korean EBS learners.
Practice target items: [${targetWords || ""}].
Rules:
1. Respond naturally in friendly English (1-3 sentences).
2. Check student's response for grammar. If needed, provide brief feedback at end as: [FEEDBACK]: (explanation).`;

    // Map messages to format expected by generateContent contents
    const contents = messages.map((m) => ({
      role: m.role === "bot" || m.role === "model" ? "model" : "user",
      parts: Array.isArray(m.parts) ? m.parts : [{ text: m.text || "" }],
    }));

    const response = await generateWithFallback({
      userApiKey,
      contents,
      systemInstruction: systemPrompt,
      preferredModel: "gemini-3.1-flash-lite",
    });

    const reply = response.text || "Sorry, I couldn't process that.";
    res.json({ success: true, text: reply });
  } catch (error) {
    console.error("Chat Error:", error);
    res.status(500).json({ error: error.message || "대화 응답 생성 중 오류가 발생했습니다." });
  }
});

// Serve static frontend
app.use(express.static(__dirname));

// Fallback to index.html
app.use((req, res) => {
  res.sendFile(path.join(__dirname, "index.html"));
});

app.listen(PORT, "0.0.0.0", () => {
  console.log(`EBS Study Mate Server running on http://0.0.0.0:${PORT}`);
});
