import time
from backend.config import EMOTION_VA, M1_W_SEC, M1_MIN_FACES, M2_W1, M2_W2, M2_R_HI, M2_R_SPIKE, M2_COOLDOWN

class ModeAggregator:
    def __init__(self):
        self.history = [] # store frame results
        self.start_time = time.time()
        self.m2_last_alarm_ts = 0
        self.m5_target = "happiness"
        self.m5_score = 0
        self.m5_combo = 0
        self.m5_target_ts = 0
        
    def add_frame(self, ts, faces):
        self.history.append({'ts': ts, 'faces': faces})
        # cleanup old history (older than 300s)
        self.history = [h for h in self.history if ts - h['ts'] < 300]
        
    def _calc_va(self, probs):
        v = sum(EMOTION_VA[e]['valence'] * p for e, p in probs.items())
        a = sum(EMOTION_VA[e]['arousal'] * p for e, p in probs.items())
        return v, a
        
    def get_output(self, mode, ts):
        if mode == "m1":
            return self._m1_output(ts)
        elif mode == "m2":
            return self._m2_output(ts)
        elif mode == "m3":
            return self._m3_output(ts)
        elif mode == "m4":
            return self._m4_output(ts)
        elif mode == "m5":
            return self._m5_output(ts)
        return {}
        
    def _m1_output(self, ts):
        # M1 食堂满意度检测
        window = [h for h in self.history if ts - h['ts'] <= M1_W_SEC]
        faces = [f for h in window for f in h['faces']]
        
        if len(faces) < M1_MIN_FACES:
            return {"status": "insufficient_samples", "satisfaction": 0, "pos": 0, "neu": 0, "neg": 0, "dominant": "N/A"}
            
        v_sum = 0
        dom_counts = {}
        for f in faces:
            v, _ = self._calc_va(f['probs'])
            v_sum += v
            dom = f['dominant']
            dom_counts[dom] = dom_counts.get(dom, 0) + 1
            
        mean_v = v_sum / len(faces)
        satisfaction = max(0, min(100, (mean_v + 1) / 2 * 100))
        
        pos_cnt = sum(1 for f in faces if f['dominant'] in ['happiness', 'surprise'])
        neu_cnt = sum(1 for f in faces if f['dominant'] == 'neutral')
        neg_cnt = len(faces) - pos_cnt - neu_cnt
        
        dom = max(dom_counts, key=dom_counts.get) if dom_counts else "N/A"
        
        return {
            "status": "ok",
            "satisfaction": satisfaction,
            "pos": pos_cnt / len(faces),
            "neu": neu_cnt / len(faces),
            "neg": neg_cnt / len(faces),
            "dominant": dom
        }
        
    def _m2_output(self, ts):
        # M2 突发情绪预警
        if not self.history: return {"r": 0, "alarm": False}
        faces = self.history[-1]['faces']
        if not faces: return {"r": 0, "alarm": False}
        
        # take the main face (e.g. largest box)
        face = max(faces, key=lambda f: f['bbox'][2]*f['bbox'][3])
        p_neg = face['probs']['anger'] + face['probs']['disgust']
        _, a = self._calc_va(face['probs'])
        
        r = M2_W1 * p_neg + M2_W2 * a
        
        alarm = False
        if ts - self.m2_last_alarm_ts > M2_COOLDOWN:
            # check spike
            past_frame = [h for h in self.history if ts - h['ts'] >= 0.4]
            if past_frame:
                past_face = max(past_frame[-1]['faces'], key=lambda f: f['bbox'][2]*f['bbox'][3], default=None)
                if past_face:
                    p_neg_past = past_face['probs']['anger'] + past_face['probs']['disgust']
                    _, a_past = self._calc_va(past_face['probs'])
                    r_past = M2_W1 * p_neg_past + M2_W2 * a_past
                    if r - r_past > M2_R_SPIKE:
                        alarm = True
                        
            # check continuous
            recent_frames = [h for h in self.history if ts - h['ts'] <= 0.5] # ~5 frames at 10fps
            if len(recent_frames) >= 5:
                is_high = True
                for h in recent_frames[-5:]:
                    if not h['faces']: 
                        is_high = False; break
                    f = max(h['faces'], key=lambda f: f['bbox'][2]*f['bbox'][3])
                    pn = f['probs']['anger'] + f['probs']['disgust']
                    _, fa = self._calc_va(f['probs'])
                    fr = M2_W1 * pn + M2_W2 * fa
                    if fr < M2_R_HI:
                        is_high = False; break
                if is_high:
                    alarm = True
                    
        if alarm:
            self.m2_last_alarm_ts = ts
            
        return {"r": r, "alarm": alarm}

    def _m3_output(self, ts):
        # M3 观影记录仪
        if not self.history: return {"v": 0, "probs": {k: 0 for k in EMOTION_VA.keys()}, "spike": None}
        faces = self.history[-1]['faces']
        if not faces: return {"v": 0, "probs": {k: 0 for k in EMOTION_VA.keys()}, "spike": None}
        
        mean_probs = {k: 0.0 for k in EMOTION_VA.keys()}
        mean_v = 0.0
        for f in faces:
            for k in mean_probs:
                mean_probs[k] += f['probs'][k]
            v, _ = self._calc_va(f['probs'])
            mean_v += v
            
        mean_probs = {k: v/len(faces) for k, v in mean_probs.items()}
        mean_v /= len(faces)
        
        spike = None
        for k, p in mean_probs.items():
            if k != 'neutral' and p > 0.7:
                spike = {'emotion': k, 'intensity': p}
                break
                
        return {"v": mean_v, "probs": mean_probs, "spike": spike}

    def _m4_output(self, ts):
        # M4 演讲教练
        session = self.history
        if not session: return {}
        
        pos_time = 0
        anx_time = 0
        neu_time = 0
        total_time = 0
        
        for h in session:
            if not h['faces']: continue
            f = h['faces'][0] # single face
            v, _ = self._calc_va(f['probs'])
            if v > 0.2: pos_time += 1
            if f['probs']['fear'] + f['probs']['sadness'] > max(f['probs'].values()) - 0.1:
                anx_time += 1
            if f['dominant'] == 'neutral':
                neu_time += 1
            total_time += 1
            
        if total_time == 0: return {}
        
        pos_ratio = pos_time / total_time
        anx_ratio = anx_time / total_time
        neu_ratio = neu_time / total_time
        expressiveness = 1.0 - neu_ratio
        
        advice = "Keep it up!"
        if anx_ratio > 0.3: advice = "Too much anxiety, slow down and breathe"
        elif expressiveness < 0.2: advice = "Expressions too flat, add more variation"
        elif pos_ratio > 0.5: advice = "Great charisma!"
        
        return {
            "pos_ratio": pos_ratio,
            "anx_ratio": anx_ratio,
            "expressiveness": expressiveness,
            "advice": advice
        }
        
    def _m5_output(self, ts):
        # M5 游戏
        import random
        if ts - self.m5_target_ts > 5:
            # new target
            self.m5_target = random.choice([k for k in EMOTION_VA.keys() if k != 'neutral'])
            self.m5_target_ts = ts
            
        faces = self.history[-1]['faces'] if self.history else []
        p_target = 0
        if faces:
            f = max(faces, key=lambda f: f['bbox'][2]*f['bbox'][3])
            p_target = f['probs'][self.m5_target]
            
            if p_target > 0.6:
                self.m5_score += 10 + self.m5_combo * 2
                self.m5_combo += 1
                self.m5_target = random.choice([k for k in EMOTION_VA.keys() if k != 'neutral' and k != self.m5_target])
                self.m5_target_ts = ts
                
        return {
            "target": self.m5_target,
            "p_target": p_target,
            "score": self.m5_score,
            "combo": self.m5_combo,
            "time_left": max(0, 5 - (ts - self.m5_target_ts))
        }
