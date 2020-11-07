import os
import threading
import tkinter as tk
import webbrowser
from datetime import datetime
from threading import Timer
from tkinter import Label, W, SUNKEN, HORIZONTAL, E, N, S, filedialog
from tkinter.ttk import Progressbar

import numpy as np
import tdt
from PIL import ImageTk, Image
from dateutil.tz.tz import tzlocal
from pynwb import NWBFile, NWBHDF5IO
from pynwb.file import Subject
from scipy.io import loadmat

from src.bsl_python.GUI.app import app
from src.bsl_python.GUI.dashboard import Dashboard
from src.bsl_python.lab_book_loader import LabBookLoader
from src.bsl_python.preprocessing.preprocess import preprocess_nwbfile


class Launcher(tk.Tk):
    def __init__(self):
        super().__init__()
        self.progress_frame = None
        self.progress_bar = None
        self.history_frame = None
        self.log = None
        self.button_frame = None
        self.content_frame = None
        self.title('BSL')
        self.resizable(width=False, height=False)
        self.create_default_layout()
        self.notebook = None
        self.path = None

    def home(self):
        self.clear()
        self.create_default_layout()

    def clear(self):
        for widget in self.winfo_children():
            widget.destroy()
        self.notebook = None
        self.path = None

    def create_default_layout(self):
        self.create_progress_bar()
        self.content_frame = tk.Frame(master=self)
        load = Image.open("res/excel.png").resize((75, 75), Image.ANTIALIAS)
        render = ImageTk.PhotoImage(load)
        img2 = Label(self.content_frame, image=render)
        img2.image = render
        img2.place(x=0, y=0)
        img2.grid(row=0, column=0)
        label = tk.Label(self.content_frame,
                         text="(Recommended)\nLoad notebook and convert data\nto universal format\n(Neurodata Without Border).")
        label.grid(row=1, column=0)
        button_load = tk.Button(master=self.content_frame,
                                text="Load Lab notebook",
                                width=25,
                                height=2,
                                command=self.load_notebook)
        button_load.grid(row=2, column=0, padx=10, pady=10)

        load = Image.open("res/dashboard.png").resize((75, 75), Image.ANTIALIAS)
        render = ImageTk.PhotoImage(load)
        img2 = Label(self.content_frame, image=render)
        img2.image = render
        img2.place(x=0, y=0)
        img2.grid(row=0, column=1)
        label = tk.Label(self.content_frame,
                         text="Data visualization: \nRaster plot, PSTH...\nRequires NWB files.")
        label.grid(row=1, column=1)
        button_dash = tk.Button(master=self.content_frame,
                                text="Open visualization",
                                width=25,
                                height=2,
                                command=lambda: visualize_data(self.path, self.notebook))
        button_dash.grid(row=2, column=1, padx=10, pady=10)

        load = Image.open("res/preprocessing.png").resize((75, 75), Image.ANTIALIAS)
        render = ImageTk.PhotoImage(load)
        img2 = Label(self.content_frame, image=render)
        img2.image = render
        img2.place(x=0, y=0)
        img2.grid(row=0, column=2)
        label = tk.Label(self.content_frame,
                         text="Compute and save analysis\nRequires NWB files.")
        label.grid(row=1, column=2)
        button_preprocess = tk.Button(master=self.content_frame,
                                      text="Pre-process data and save",
                                      width=25,
                                      height=2,
                                      command=lambda: self.preprocess_data())
        button_preprocess.grid(row=2, column=2, padx=10, pady=10)
        self.content_frame.grid(row=0, column=0, pady=10)

    def create_notebook_loading_layout(self):
        self.create_progress_bar()
        self.history_frame = tk.Frame(master=self)
        button_back = tk.Button(master=self.history_frame,
                                text="Back",
                                width=8,
                                height=1,
                                command=self.home)
        button_back.grid(row=0, column=0, padx=10, pady=5, sticky=W)
        self.history_frame.grid(row=0, column=0, pady=10)

        self.content_frame = tk.Frame(master=self)
        button_load = tk.Button(master=self.content_frame,
                                text="Load Lab notebook",
                                width=25,
                                height=2,
                                command=self.load_notebook)
        button_load.grid(row=0, column=0, padx=10, pady=10)
        self.content_frame.grid(row=1, column=0, pady=10)

    def create_progress_bar(self):
        self.progress_frame = tk.Frame(master=self, relief=SUNKEN, borderwidth=1)
        self.progress_bar = Progressbar(master=self.progress_frame, orient=HORIZONTAL, length=100, mode='determinate')
        self.log = tk.Label(self.progress_frame, text="Progress:")
        self.progress_bar.grid(row=0, column=0, sticky="W")
        self.log.grid(row=0, column=1, columnspan=2, sticky="W")
        self.progress_frame.grid(row=2, sticky=W + E + N + S, columnspan=2)

    def show_data_status(self, stimulation_status, clustering_status):
        frame_files = tk.Frame(master=self.content_frame)
        load = Image.open("res/clustering.png").resize((75, 75), Image.ANTIALIAS)
        render = ImageTk.PhotoImage(load)
        img = Label(frame_files, image=render)
        label_text = "Spike sorted data"
        if clustering_status == 2:
            label_text += "\nmissing blocks"
        elif clustering_status == 0:
            label_text += "\nno data found"
        label_img = Label(frame_files, text=label_text)
        img.image = render
        img.place(x=0, y=0)
        load = Image.open("res/stimulation.png").resize((75, 75), Image.ANTIALIAS)
        render = ImageTk.PhotoImage(load)
        img2 = Label(frame_files, image=render)
        label_text = "Stimulation parameters"
        if stimulation_status == 2:
            label_text += "\nmissing blocks"
        elif stimulation_status == 0:
            label_text += "\nno data found"
        label_img2 = Label(frame_files, text=label_text)
        img2.image = render
        img2.place(x=0, y=0)

        canvas_before_top, canvas_before_bottom = self.create_diverging_arrows(clustering_status, frame_files,
                                                                               stimulation_status)

        if stimulation_status > 0 and clustering_status > 0:
            width_canvas = 100
            height_canvas = 100
            canvas = tk.Canvas(frame_files, width=width_canvas, height=height_canvas)
            color = "#FF0000"
            if clustering_status == 1:
                color = "#32CD32"
            elif clustering_status == 2:
                color = "#FF7F50"
            canvas.create_line(0, height_canvas / 3, width_canvas, height_canvas, arrow=tk.LAST, width=2.5, fill=color)
            canvas_bottom = tk.Canvas(frame_files, width=width_canvas, height=height_canvas)
            color = "#FF0000"
            if stimulation_status == 1:
                color = "#32CD32"
            elif stimulation_status == 2:
                color = "#FF7F50"
            canvas_bottom.create_line(0, height_canvas * 2 / 3, width_canvas, 0, arrow=tk.LAST, width=2.5, fill=color)
            canvas.grid(row=1, column=2)
            canvas_bottom.grid(row=2, column=2)

        img.grid(row=1, column=1)
        label_img.grid(row=0, column=1)
        img2.grid(row=2, column=1)
        label_img2.grid(row=3, column=1)
        canvas_before_top.grid(row=1, column=0)
        canvas_before_bottom.grid(row=2, column=0)

        frame_files.grid(row=0, column=1, padx=10, pady=10)

    def create_diverging_arrows(self, arrow_status_top, frame_files, arrow_status_bottom):
        width_canvas = 100
        height_canvas = 100
        canvas_before_top = tk.Canvas(frame_files, width=width_canvas, height=height_canvas)
        color = "#FF0000"
        if arrow_status_top == 1:
            color = "#32CD32"
        elif arrow_status_top == 2:
            color = "#FF7F50"
        canvas_before_top.create_line(0, height_canvas, width_canvas, height_canvas / 3, arrow=tk.LAST, width=2.5,
                                      fill=color)
        canvas_before_bottom = tk.Canvas(frame_files, width=width_canvas, height=height_canvas)
        color = "#FF0000"
        if arrow_status_bottom == 1:
            color = "#32CD32"
        elif arrow_status_bottom == 2:
            color = "#FF7F50"
        canvas_before_bottom.create_line(0, 0, width_canvas, height_canvas * 2 / 3, arrow=tk.LAST, width=2.5,
                                         fill=color)
        return canvas_before_top, canvas_before_bottom

    def show_buttons(self, compact_status):
        self.button_frame = tk.Frame(master=self.content_frame)
        load = Image.open("res/nwb.png").resize((125, 50), Image.ANTIALIAS)
        render = ImageTk.PhotoImage(load)
        img = Label(self.button_frame, image=render)
        img.image = render
        img.place(x=0, y=0)
        button_compact = tk.Button(master=self.button_frame,
                                   text="Compact data",
                                   width=25,
                                   height=2,
                                   command=self.compact_data)
        canvas_before_top, canvas_before_bottom = self.create_diverging_arrows(compact_status, self.button_frame,
                                                                               compact_status)
        button_visualize = tk.Button(master=self.button_frame,
                                     text="Visualize and save",
                                     width=25,
                                     height=2,
                                     command=lambda: visualize_data(self.path, self.notebook),
                                     state="disabled" if compact_status == 0 else "normal")
        button_preprocess = tk.Button(master=self.button_frame,
                                      text="Pre-process and save",
                                      width=25,
                                      height=2,
                                      command=lambda: self.preprocess_data(),
                                      state="disabled" if compact_status == 0 else "normal")
        img.grid(row=0, column=0)
        button_compact.grid(row=1, column=0, padx=10)
        canvas_before_top.grid(row=0, column=1)
        canvas_before_bottom.grid(row=2, column=1)
        button_visualize.grid(row=0, column=2, padx=10)
        button_preprocess.grid(row=2, column=2, padx=10, pady=20)
        self.button_frame.grid(row=0, column=2, padx=10, pady=10)

    def load_notebook(self):
        self.clear()
        self.create_notebook_loading_layout()
        info, path = select_notebook()
        self.notebook = info
        self.path = path

        data_path = os.path.join(path, info["Experiment"]["ID"])
        block_names = os.listdir(data_path)
        list_block_stim = []
        for block in block_names:
            full_path_tdt = os.path.join(data_path, block)
            if os.path.isdir(full_path_tdt) and "Block" in block:
                block_number = int(block.split('-')[1])
                list_block_stim.append(block_number)
        list_block_stim = sorted(list_block_stim)
        stim_status = 0
        if list_block_stim == info["Trials"]["Block"]:
            stim_status = 1
        elif len(list_block_stim) > 0:
            stim_status = 2
        list_block_sorted = []
        block_names = os.listdir(os.path.join(data_path, "kwik2"))
        for block in block_names:
            block_name = block.split("_")[1][0:-4]
            full_path_kwik2 = os.path.join(data_path, "kwik2", block)
            if os.path.exists(full_path_kwik2):
                block_number = int(block_name.split('-')[1])
                list_block_sorted.append(block_number)
        list_block_sorted = sorted(list_block_sorted)
        clustering_status = 0
        if list_block_sorted == info["Trials"]["Block"]:
            clustering_status = 1
        elif len(list_block_sorted) > 0:
            clustering_status = 2
        self.show_data_status(stim_status, clustering_status)

        path_to_nwb = os.path.join(self.path, self.notebook["Experiment"]["ID"], "NWB")
        compact_status = 0
        if os.path.exists(path_to_nwb):
            block_names = [file for file in os.listdir(path_to_nwb) if ".nwb" in file]
            list_block_compacted = []
            for block in block_names:
                block_number = int(block.split('-')[1][0:-4])
                list_block_compacted.append(block_number)
            list_block_compacted = sorted(list_block_compacted)
            if list_block_compacted == info["Trials"]["Block"]:
                compact_status = 1
            elif len(list_block_compacted) > 0:
                compact_status = 2
        if stim_status > 0 and clustering_status > 0:
            self.show_buttons(compact_status)

    def compact_data(self):
        self.reset_progress_bar()
        self.update_progress_bar(0, "Load stimulation data")
        data = load_stimulation_data(os.path.join(self.path, self.notebook["Experiment"]["ID"]))
        self.update_progress_bar(20, "Load sorted spike data")
        data2 = load_spikes_data(os.path.join(self.path, self.notebook["Experiment"]["ID"]))
        self.update_progress_bar(40, "Compact blocks")
        for block in data2:
            if block in data:
                self.compact_block(block, data, data2)
        list_block_compacted = data2.keys()
        list_block_compacted = sorted(list_block_compacted)
        compact_status = 0
        if list_block_compacted == self.notebook["Trials"]["Block"]:
            compact_status = 1
        elif len(list_block_compacted) > 0:
            compact_status = 2
        self.show_buttons(compact_status)
        self.update_progress_bar(100, "Done")

    def reset_progress_bar(self):
        self.progress_bar['value'] = 0
        self.log.config(text='')
        self.update_idletasks()

    def update_progress_bar(self, added_value, text):
        self.progress_bar['value'] += added_value
        self.log.config(text=text)
        self.update()

    def preprocess_data(self):
        nwb_path = os.path.join(self.path, self.notebook["Experiment"]["ID"], "NWB")
        list_files = os.listdir(nwb_path)
        self.update_progress_bar(0, "Preprocessing data")
        errors = 0
        for index in range(len(list_files)):
            if ".nwb" in list_files[index]:
                try:
                    preprocess_nwbfile(nwb_path, list_files[index])
                    self.update_progress_bar(index / len(list_files), "Preprocessing data: " + list_files[
                        index] + " - Done. Error encountered: " + str(errors))
                except NameError as e:
                    errors += 1
                    print("Error with file " + list_files[index] + ". " + str(e))
                    self.update_progress_bar(index / len(list_files),
                                             "Preprocessing data: " + list_files[index] + " - Error")
        self.update_progress_bar(1, "Pre-processing done. Error encountered: " + str(errors))

    def compact_block(self, block, data, data2):
        print("Save block " + str(block))
        self.update_progress_bar(round(40 / len(data2)), "Save block " + str(block))
        subject = Subject(age=str(self.notebook["Mouse"]["Age"]), genotype=self.notebook["Mouse"]["Strain"],
                          species="Mouse",
                          subject_id=self.notebook["Mouse"]["ID"], sex=self.notebook["Mouse"]["Gender"],
                          weight=str(self.notebook["Mouse"]["Weight"]),
                          date_of_birth=self.notebook["Mouse"]["DateOfBirth"])
        nwbfile = NWBFile('', identifier=str(block), session_start_time=datetime.now(tzlocal()),
                          experimenter=self.notebook["Experiment"]["Experimenter"],
                          lab='Brain Sound Lab',
                          institution='University of Basel',
                          experiment_description='',
                          session_id=self.notebook["Experiment"]["ID"],
                          subject=subject,
                          protocol=self.notebook["Trials"]["StimulusSet"][block - 1])
        device = nwbfile.create_device(name="TDT")
        cortical_region = str(list(np.unique(self.notebook["Electrophy"]["Cortical region"]))[0])
        electrode_group = nwbfile.create_electrode_group(
            self.notebook["Electrophy"]["Electrode ID"].replace('/', ''),
            description=self.notebook["Electrophy"]["Electrode Type"],
            location=cortical_region,
            device=device)
        for index in range(self.notebook["Electrophy"]["Nb channels"]):
            # TODO: Replace index and position by electrode configuration
            nwbfile.add_electrode(id=int(index),
                                  x=1.0, y=2.0, z=3.0,
                                  imp=float(-index),
                                  location=cortical_region, filtering='none',
                                  group=electrode_group)
        nb_trials = 0
        trial_key = ""
        for key in data[block]["epocs"].keys():
            if key == "coun" or key == "ChnA":
                attribute = data[block]["epocs"][key]
                nb_trials = len(attribute["data"])
                trial_key = key
        for key in data[block]["epocs"].keys():
            attribute = data[block]["epocs"][key]
            if len(attribute["data"]) == nb_trials:
                nwbfile.add_trial_column(name=key, description='Experiment trial')

        clusters = list(data2[block]["waveforms"]["clusters"][0])
        for trial in range(nb_trials):
            trial_parameters = {}
            for key in data[block]["epocs"].keys():
                attribute = data[block]["epocs"][key]
                if len(attribute["data"]) == nb_trials:
                    trial_parameters[key] = attribute["data"][trial]
            nwbfile.add_trial(start_time=float(data[block]["epocs"][trial_key]["onset"][trial]),
                              stop_time=float(data[block]["epocs"][trial_key]["offset"][trial]),
                              **trial_parameters)
        index = 0
        for cluster in clusters:
            indices = data2[block]["spikes"][:, 1] == cluster
            if len(indices) > 0:
                spike_times = [x for x in list(data2[block]["spikes"][indices, 0])]
                waveform_sd = data2[block]["waveforms"]["std"][0, index].transpose().tolist()[0]
                waveform_mean = list(data2[block]["waveforms"]["waveforms"][0, index][0])
                electrodes = [x - 1 for x in list(np.unique(data2[block]["spikes"][indices, 3]))]
                index += 1
                nwbfile.add_unit(spike_times=spike_times,
                                 waveform_sd=waveform_sd,
                                 waveform_mean=waveform_mean,
                                 electrodes=electrodes,
                                 id=int(cluster))
        path_to_nwb = os.path.join(self.path, self.notebook["Experiment"]["ID"], "NWB")
        if not os.path.exists(path_to_nwb):
            os.mkdir(path_to_nwb)
        io = NWBHDF5IO(
            os.path.join(path_to_nwb, self.notebook["Experiment"]["ID"] + "_Block-" + str(block) + '.nwb'),
            mode='w')
        try:
            io.write(nwbfile)
        except:
            pass
        io.close()


def open_browser(port):
    webbrowser.open_new("http://localhost:{}".format(port))


def select_notebook():
    initial_dir = "/"
    path_file = ".path"
    if os.path.exists(path_file):
        with open(path_file, "r") as f:
            path = f.readline()
            if os.path.isdir(path):
                initial_dir = path
    fullpath = filedialog.askopenfilename(initialdir=initial_dir, title="Select file",
                                          filetypes=(("notebook files", "*.xlsx"), ("all files", "*.*")))
    if fullpath == "":
        raise ValueError("No file selected")
    path = os.path.dirname(fullpath)
    with open(path_file, "w") as f:
        f.write(path)
    filename = os.path.basename(fullpath)
    info = LabBookLoader().load_notebook(path, filename)
    return info, path


def load_stimulation_data(path):
    all_data = {}
    block_names = os.listdir(path)
    for block in block_names:
        full_path_tdt = os.path.join(path, block)
        if os.path.isdir(full_path_tdt) and "Block" in block:
            data = tdt.read_block(full_path_tdt, evtype=['epocs', 'snips', 'scalars'], nodata=1)
            block_number = int(block.split('-')[1])
            all_data[block_number] = {"epocs": data.epocs, "snips": data.snips, "eNeu": data.snips.eNeu}
    return all_data


def load_spikes_data(path):
    all_data = {}
    block_names = os.listdir(os.path.join(path, "kwik2"))
    for block in block_names:
        block_name = block.split("_")[1][0:-4]
        full_path_kwik2 = os.path.join(path, "kwik2", block)
        data = loadmat(full_path_kwik2)
        block_number = int(block_name.split('-')[1])
        all_data[block_number] = {"spikes": data["spikes"], "waveforms": data["waveforms"][0, 0]}
    return all_data


def launch_server(path, notebook):
    port = 8050
    Timer(1.5, open_browser, [port]).start()
    nwb_path = os.path.join(path, notebook["Experiment"]["ID"], "NWB")
    nwbfiles = {"Block " + (file.split('-')[-1].replace('.nwb', '')).zfill(2): os.path.join(nwb_path, file) for file in
                os.listdir(nwb_path) if ".nwb" in file}
    c_file = list(nwbfiles.keys())[0]
    nwb_io = NWBHDF5IO(nwbfiles[c_file], 'r')
    nwbfile = nwb_io.read()

    Dashboard.create(nwbfile, nwbfiles, notebook)
    app.run_server(debug=False, port=port)


def visualize_data(path, notebook):
    x = threading.Thread(target=launch_server, args=(path, notebook))
    x.start()
