import torch
import torch.nn as nn

# from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
from transformers import AutoProcessor, AutoModelForCausalLM
import numpy as np
from decord import VideoReader, cpu
from PIL import Image

def get_frames_in_interval(start, end, number_of_frames):
        indices = np.random.randint(low = start, high = end, size = number_of_frames)
        indices = np.sort(indices)
        return indices


def sample_frames_on_interval(file_path, num_frames):
    

    def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
        # Making three intervals: Start, middle and the end
        interval_gap = seg_len // 3
        frames_per_gap = clip_len // 3
        
        begin_interval = get_frames_in_interval(0, interval_gap, frames_per_gap)
        middle_interval = get_frames_in_interval(interval_gap + 1, interval_gap * 2, frames_per_gap)
        end_interval = get_frames_in_interval((interval_gap*2) + 1, seg_len, frames_per_gap)
        frame_indices = np.concatenate((begin_interval, middle_interval, end_interval), axis = 0)
        # print(frame_indices)
        return frame_indices
    
    # video clip consists of 300 frames (10 seconds at 30 FPS)
    try:
        videoreader = VideoReader(file_path, num_threads=1, ctx=cpu(0))
    except:
        print("******"*10)
        print("Failed for : ", file_path)
    # sample 6 frames
    videoreader.seek(0)
    indices = sample_frame_indices(clip_len=num_frames, frame_sample_rate=4, seg_len=len(videoreader))
    frames = videoreader.get_batch(indices).asnumpy()

    return list(frames)

    

def sample_all_frames_on_multiple_interval(file_path, num_frames, interval_count=4):
    

    def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
        # Making 4 interval of videos
        no_of_intervals = interval_count
        interval_gap = seg_len // no_of_intervals
        frames_per_gap = 6

        frames = []

        for i in range(0, no_of_intervals):
            curr_interval = get_frames_in_interval(interval_gap * i, interval_gap * (i+1), frames_per_gap)
            frames.append(curr_interval)
        return frames
    


    # video clip consists of 300 frames (10 seconds at 30 FPS)
    videoreader = VideoReader(file_path, num_threads=1, ctx=cpu(0))

    videoreader.seek(0)
    indices = sample_frame_indices(clip_len=num_frames, frame_sample_rate=4, seg_len=len(videoreader))
    frame_set = []
    for index_set in indices:
        video_frames = videoreader.get_batch(index_set).asnumpy()
        frame_set.append(list(video_frames))

    return frame_set


def make_video_from_image(file_path, num_frames):
    image = Image.open(file_path).convert('RGB')
    image.load()
    data = np.asarray(image)
    # print(data.shape)
    frames = [data] * num_frames
    return frames




class GITVideoTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        print("")

        self.is_loaded = False

        # vision_tower_name = "microsoft/git-large-vatex"
        self.vision_tower_name = vision_tower
        # self.select_layer = args.mm_vision_select_layer
        self.select_layer = -2
        try:
            self.interval_type = args.interval_type
            self.no_of_intervals = args.no_of_intervals
        except:
            self.interval_type = "multi_interval"
            self.no_of_intervals = 4
        


        self.load_model()
        # if not delay_load:
            
        # else:
        #     self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)
        #     # raise("Not implemented!!")

    def load_model(self):
        self.processor = AutoProcessor.from_pretrained(self.vision_tower_name, local_files_only=True)
        self.image_processor = AutoProcessor.from_pretrained(self.vision_tower_name,  local_files_only=True)
        self.vision_tower = AutoModelForCausalLM.from_pretrained(self.vision_tower_name, output_hidden_states=True,  local_files_only=True)
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    @torch.no_grad()
    def featurize_video(self, video_path):
        
        
        # # 6 frames
        # combine_multi_interval = False
        # # 18 frames
        # # combine_multi_interval = True
        
        num_frames = self.config.num_image_with_embedding
        
        
    
        input_ids = [self.processor.tokenizer.cls_token_id]
        input_ids = torch.tensor(input_ids).unsqueeze(0)
        input_ids = input_ids.to(self.device)

        if self.interval_type == "single":
            frames = sample_frames_on_interval(video_path, num_frames)#.to(self.device)
            inputs = self.processor(images=frames, return_tensors="pt")#.to(self.dtype).to(self.device)
            outputs = self.vision_tower(input_ids, pixel_values=inputs.pixel_values.to(self.device).to(self.dtype))
            features = outputs.hidden_states[self.select_layer].to(self.dtype)
        elif self.interval_type == "image_combination":
            frames = make_video_from_image(video_path, num_frames)
            inputs = self.processor(images=frames, return_tensors="pt")#.to(self.dtype).to(self.device)
            outputs = self.vision_tower(input_ids, pixel_values=inputs.pixel_values.to(self.device).to(self.dtype))
            features = outputs.hidden_states[self.select_layer].to(self.dtype)
        else:
            all_outputs = []
            # print("\n\n\n\n\n")
            # print("*"*20)
            # print("multiple intervals: ", self.no_of_intervals)
            # print("*"*20)
            frames = sample_all_frames_on_multiple_interval(video_path, num_frames, self.no_of_intervals)#.to(self.device)
            for frame in frames:
                inputs = self.processor(images=frame, return_tensors="pt")#.to(self.dtype).to(self.device)
                with torch.no_grad():
                    outputs = self.vision_tower(input_ids, pixel_values=inputs.pixel_values.to(self.device).to(self.dtype))
                    hidden_states = outputs.hidden_states[self.select_layer].to(self.dtype)
                    all_outputs.append(hidden_states)
            # all_features = all_outputs[0] + all_outputs[1] + all_outputs[2] + all_outputs[3] + all_outputs[4] + all_outputs[5] + all_outputs[6] + all_outputs[7]
            # all_features =
            # all_features = sum(all_outputs)
            all_features = torch.stack(all_outputs).sum(dim=0)
            # all_features = all_outputs[0] + all_outputs[1] + all_outputs[2] + all_outputs[3] + all_outputs[4] + all_outputs[5]
            # all_features = torch.cat(all_outputs, dim=0)
            features = all_features
            # features = torch.div(all_features, 3)
            

        return features

    @torch.no_grad()
    def forward(self, video_paths):
        # print(type(video_paths), video_paths)
        if type(video_paths) is list:
            features = []
            for path in video_paths:
                current_video_feature = self.featurize_video(path)
                features.append(current_video_feature)
            features = torch.cat(features).to(self.dtype)
            return features
        else:
            return self.featurize_video(video_paths)

        # num_frames = self.config.num_image_with_embedding
        # print("**"*10)
        # print(video_path)
        # frames = sample_frames(video_path[0], num_frames)#.to(self.device)
        # inputs = self.processor(images=frames, return_tensors="pt").to(self.device).to(self.dtype)
        # question = "The metaphor in the video is: "

        # input_ids = self.processor(text=question, add_special_tokens=False).input_ids
        # input_ids = [self.processor.tokenizer.cls_token_id] + input_ids
        # input_ids = torch.tensor(input_ids).unsqueeze(0)
        # input_ids = input_ids.to(self.device)
        # outputs = self.vision_tower(input_ids, pixel_values=inputs.pixel_values).to(self.device).to(self.dtype)
        # features = outputs.hidden_states[self.select_layer].to(self.dtype)
        # return features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype
        # return torch.bfloat16
        # return torch.float32

    @property
    def return_dtype(self):
        return torch.bfloat16

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        # print("***"*20)
        # print("Config: ", self.vision_tower.config)
        # print("***"*20)

        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2
