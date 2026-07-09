#pragma once

constexpr int kAgingChannels = 9;
constexpr float kRefChannelMax[9] = {0.849386811f, 0.835293651f, 0.809992254f, 0.825033247f, 0.776530504f, 0.80995357f, 0.830552697f, 0.824463844f, 0.818351865f};
constexpr float kAgingLeak = 0.99993f;
constexpr float kAgingGainMin = 1.0f;
constexpr float kAgingGainMax = 6.0f;
