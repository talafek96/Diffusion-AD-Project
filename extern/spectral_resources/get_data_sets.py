import math
from scipy.signal import correlate
from scipy.stats.stats import spearmanr
from scipy.stats.stats import pearsonr
from matplotlib.widgets import PolygonSelector
from spectral import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
from matplotlib.patches import Polygon
import pickle
from scipy.signal import find_peaks
from load_and_crop_img import HDR_PATH_SWIR as HDR_PATH, IMAGE_PATH_SWIR as IMAGE_PATH

# SWIR
# HDR_PATH = r"C:\projects_files\hyper_spectral\Gvaram_17.05.2023\Gvaram_50m\swir\batch process\100083_gvaram_17_05_2023_50m\raw_11700_rdk_rd_rf.hdr"
# IMAGE_PATH = r"C:\projects_files\hyper_spectral\Gvaram_17.05.2023\Gvaram_50m\swir\batch process\100083_gvaram_17_05_2023_50m\raw_11700_rdk_rd_rf"
VNIR = False

# #VNIR
# HDR_PATH = r"C:\projects_files\hyper_spectral\Gvaram_17.05.2023\Gvaram_50m\vnir\batch process\100082_gvaram_17_05_2023_50m\raw_11216_rd_rf.hdr"
# IMAGE_PATH = r"C:\projects_files\hyper_spectral\Gvaram_17.05.2023\Gvaram_50m\vnir\batch process\100082_gvaram_17_05_2023_50m\raw_11216_rd_rf"
# VNIR = True

HDR = envi.open(HDR_PATH, IMAGE_PATH)
# HDR_LOAD = HDR.load()
wvl = HDR.bands.centers
ROWS, COLS, BANDS = HDR.nrows, HDR.ncols, HDR.nbands
FLG_EXIT = False
V_FABRIC = [0.0498801, 0.0518286, 0.0556343, 0.0578088, 0.0579851, 0.0564507, 0.0583113, 0.0653091, 0.0645667,
            0.0651354, 0.0642886, 0.0629095, 0.0608354, 0.0582653, 0.0584261, 0.0571198, 0.056016, 0.0554591, 0.0553269,
            0.0551452, 0.0549163, 0.0546865, 0.0545974, 0.0543445, 0.0542391, 0.0541198, 0.0541437, 0.0541792,
            0.0543384, 0.0543494, 0.0542989, 0.0543505, 0.0545878, 0.0535303, 0.0558046, 0.0573238, 0.0585698,
            0.0615287, 0.0673687, 0.0688452, 0.0657481, 0.0644857, 0.0674492, 0.0669969, 0.0637172, 0.0605063,
            0.0590406, 0.0583999, 0.0586843, 0.0594734, 0.0593979, 0.0599514, 0.0601806, 0.0598641, 0.0597702,
            0.0593395, 0.058876, 0.0583021, 0.058129, 0.0581202, 0.0583403, 0.0589133, 0.060025, 0.0599409, 0.0587959,
            0.0583392, 0.0582384, 0.0588488, 0.0597144, 0.0607744, 0.0622177, 0.0634471, 0.0647056, 0.0683607,
            0.0693214, 0.0707624, 0.078514, 0.0857122, 0.0843108, 0.090457, 0.0921614, 0.0952308, 0.0954711, 0.098843,
            0.0985062, 0.0980644, 0.1002238, 0.1016386, 0.0989043, 0.0996272, 0.0984019, 0.0951133, 0.0933897,
            0.0921349, 0.0902414, 0.0890895, 0.0875228, 0.0864066, 0.085072, 0.0840556, 0.0828589, 0.081903, 0.080734,
            0.079972, 0.0792727, 0.0785921, 0.078228, 0.0776492, 0.0764116, 0.075323, 0.0745317, 0.073809, 0.0723479,
            0.0705532, 0.0686288, 0.06625, 0.0625497, 0.0575114, 0.0521304, 0.047942, 0.0462378, 0.0477186, 0.0505967,
            0.0533763, 0.0559817, 0.058356, 0.0584588, 0.0583409, 0.0581075, 0.0585509, 0.0593958, 0.0603574, 0.0616902,
            0.0635619, 0.0650941, 0.0662534, 0.0674812, 0.0688807, 0.0707953, 0.073132, 0.0739267, 0.0759558, 0.0794127,
            0.0824498, 0.0875558, 0.1066353, 0.1114359, 0.1289838, 0.1264574, 0.1173182, 0.1049094, 0.0979166,
            0.0958348, 0.0942961, 0.0942776, 0.0953552, 0.0947886, 0.0913892, 0.088876, 0.0837474, 0.0799734, 0.0784325,
            0.0771922, 0.0759897, 0.0740735, 0.0697586, 0.0655281, 0.0598355, 0.0557715, 0.053197, 0.0545453, 0.0566848,
            0.0574123, 0.0557089, 0.0568303, 0.0592526, 0.0595845, 0.058916, 0.0593572, 0.0614876, 0.0628483, 0.0624533,
            0.0607015, 0.0577833, 0.0536298, 0.0489405, 0.0440596, 0.0388021, 0.0346722, 0.03425, 0.0347857, 0.0361724,
            0.0380232, 0.0389497, 0.0413691, 0.044127, 0.0437352, 0.0430451, 0.0425304, 0.0413088, 0.0418233, 0.038986,
            0.0391518, 0.0401115, 0.0424043, 0.0430896, 0.044703, 0.0399574, 0.0469508, 0.0475601, 0.0502543, 0.0511375,
            0.0504858, 0.0516442, 0.0545039]

FABRIC_PEAKS = {7: 934,
                10: 952,
                17: 994,
                39: 1125,
                40: 1131,
                43: 1149,
                96: 1466,
                129: 1664,
                208: 2137,
                219: 2203,
                228: 2256}

COLORS = {0: (255, 0, 0),
          1: (0, 255, 0),
          2: (0, 0, 255),
          3: (255, 255, 0),
          4: (0, 255, 255)}

THE_BLACK_DICT = {76: 1347,
                  77: 1353,
                  78: 1359,
                  79: 1365,
                  80: 1371,
                  81: 1377,
                  82: 1383,
                  83: 1389,
                  84: 1395,
                  85: 1401,
                  86: 1407,
                  154: 1814,
                  155: 1820,
                  156: 1826,
                  157: 1832,
                  158: 1838,
                  159: 1843,
                  160: 1849,
                  161: 1855,
                  162: 1861,
                  163: 1867,
                  164: 1873,
                  165: 1879,
                  166: 1885,
                  167: 1891,
                  168: 1897,
                  169: 1903,
                  170: 1909,
                  171: 1915,
                  172: 1921,
                  173: 1927,
                  174: 1933,
                  175: 1939,
                  176: 1945,
                  177: 1951,
                  178: 1957,
                  179: 1963,
                  180: 2005,
                  181: 2011,
                  182: 2017,
                  253: 2406,
                  254: 2412,
                  255: 2418,
                  256: 2424,
                  257: 2430,
                  258: 2436,
                  259: 2442,
                  260: 2448,
                  261: 2454,
                  262: 2460,
                  263: 2466,
                  264: 2472,
                  265: 2478,
                  266: 2484,
                  267: 2490,
                  268: 2496,
                  269: 2502}

WVL = [x for x in wvl if math.floor(x) not in list(THE_BLACK_DICT.values())]


def on_key(event):
    if event.key == 'e':
        global FLG_EXIT
        FLG_EXIT = True
        plt.close()


def on_press(event):
    if event.button is MouseButton.RIGHT:
        graph = plt.figure(figsize=(5, 5))
        graph.canvas.manager.set_window_title("Spectral Signature")
        plt.title("x = " + str(int(event.xdata)) + ", y = " + str(int(event.ydata)))
        pixel = get_pixel(int(event.xdata), int(event.ydata))
        # plt.plot(WVL, pixel)
        plt.plot(WVL, pixel,'.')
        plt.xlabel('Wavelength')
        plt.ylabel('Reflectance')
        graph.show()


def remove_atmospheric_absorption_bands(bands):
    return np.delete(bands, list(THE_BLACK_DICT.keys()))


def data(x1, y1, x2, y2):
    arr = []
    for row in range(x1, x2):
        for col in range(y1, y2):
            arr.append(get_pixel(col, row))

    arr = np.array(arr)
    np.random.shuffle(arr)

    return arr


def tags_1(times):
    tags_arr = []
    for i in range(times):
        tags_arr.append(1)

    return np.array(tags_arr)


def tags_0(times):
    tags_arr = []
    for i in range(times):
        tags_arr.append(0)

    return np.array(tags_arr)


def get_pixel(x, y):
    if VNIR:
        pixel = HDR.read_pixel(y, x)
    else:
        pixel = remove_atmospheric_absorption_bands(HDR.read_pixel(y, x))

    return pixel


def compare_data(x1, y1, x2, y2):
    data_to_compare = {}
    for row in range(x1, x2):
        for col in range(y1, y2):
            data_to_compare[row, col] = get_pixel(col, row)

    return data_to_compare


def get_wavelengths():
    return WVL


def show_img(HDR):
    rgb = np.stack([HDR.read_band(119), HDR.read_band(52),
                    HDR.read_band(26)], axis=-1)
    # rgb = rgb / rgb.max() #* 1.5

    figure = plt.figure()
    figure.canvas.manager.set_window_title("RGB Image")
    plt.connect('button_press_event', on_press)
    plt.imshow(rgb)
    plt.show()

    return rgb


def show_bands(list_of_bands,HDR):
    if len(list_of_bands) == 1:
        rgb = np.array(HDR.read_band(list_of_bands[0]))
    else:
        rgb = np.stack([HDR.read_band(list_of_bands[0]), HDR.read_band(list_of_bands[1]),
                        HDR.read_band(list_of_bands[2])], axis=-1)
    # rgb = rgb / rgb.max() * 1.5

    figure = plt.figure()
    figure.canvas.manager.set_window_title("RGB Image")
    plt.connect('button_press_event', on_press)
    plt.imshow(rgb)
    plt.show()
    return rgb


def view_pixel_graph(x, y):
    pixel = get_pixel(x, y)
    graph = plt.figure(figsize=(5, 5))
    graph.canvas.manager.set_window_title("Spectral Signature")
    plt.title("x = " + str(int(x)) + ", y = " + str(int(y)))
    plt.plot(WVL, pixel)
    plt.xlabel('Wavelength')
    plt.ylabel('Reflectance')
    plt.show()


def view_graphs_of_image():
    graph = plt.figure()
    graph.canvas.manager.set_window_title("Spectral Signature")
    for x in range(ROWS):
        for y in range(COLS):
            pixel = get_pixel(x, y)
            plt.title("x = " + str(int(x)) + ", y = " + str(int(y)))
            plt.plot(WVL, pixel)
            plt.xlabel('Wavelength')
            plt.ylabel('Reflectance')
            plt.show()


def get_hdr():
    return HDR


# def get_hdr_load():
#     return HDR_LOAD


def insert_label(arr, num, x1, y1, x2, y2):
    for i in range(y1, y2):
        for j in range(x1, x2):
            arr[i][j] = num

    return arr


def get_bands():
    bands = []
    for x in range(ROWS):
        for y in range(COLS):
            bands.append(get_pixel(x, y))

    return np.array(bands)


def get_image():
    rgb = np.stack([HDR.read_band(119), HDR.read_band(52),
                    HDR.read_band(26)], axis=-1)
    if VNIR:
        rgb = rgb / rgb.max() * 2.5
        rgb = np.fliplr(rgb)
    else:
         rgb = rgb / rgb.max() * 20.5

    return rgb


def set_polygon():
    rgb = get_image()
    fig, ax = plt.subplots()
    fig.canvas.manager.set_window_title("Polygon Selector")
    selector = PolygonSelector(ax, lambda *args: None)
    fig.canvas.mpl_connect('key_press_event', on_key)
    fig.tight_layout()
    ax.imshow(rgb)
    plt.show()
    return selector.verts


def view_polygon(list_of_points):
    rgb = get_image()
    points = np.array(list_of_points)
    polygon = Polygon(points)
    polygon.set_facecolor('red')

    fig, ax = plt.subplots()
    fig.canvas.manager.set_window_title("Polygon")
    ax.add_patch(polygon)
    ax.imshow(rgb)
    plt.show()



def get_all_points_inside_polygon(polygon):
    points = {}
    path = polygon.get_path()
    xmin, ymin, xmax, ymax = path.get_extents().extents
    for x in range(math.floor(xmin), math.floor(xmax) + 1):
        for y in range(math.floor(ymin), math.floor(ymax) + 1):
            if polygon.contains_point((x, y)):
                points[(x, y)] = get_pixel(x, y)

    return points


def multiple_polygons():
    points_in_polygon = {}
    while FLG_EXIT is False:
        points = set_polygon()
        if len(points) > 0:
            polygon = Polygon(np.array(points))
            points_in_polygon.update(get_all_points_inside_polygon(polygon))

    return points_in_polygon


def classification_array(list_of_locations):
    new_data = np.zeros((ROWS, COLS, 3))
    for i, l in enumerate(list_of_locations):
        for location in l:
            new_data[location[0], location[1]] = COLORS[i]

    return new_data


def create_data(number_of_objects):
    list_of_locations = []
    for obj in range(1, number_of_objects + 1):
        points = multiple_polygons()
        list_of_locations.append(points.keys())
        global FLG_EXIT
        FLG_EXIT = False

        with open("object number " + str(obj) + ".txt", "wb") as f:
            pickle.dump(points, f)
            # for key, value in points.items():
            #     f.write('%s:%s\n' % (key, value))


def read_data(list_of_files_names):
    points = {}
    for files_names in list_of_files_names:
        with open(files_names, "rb") as f:
            data_points = f.read()

        points.update(pickle.loads(data_points))

    return points


def compare_vectors(v, v_to_compare):
    d = np.dot(v_to_compare, v)
    try:
        rad_angle = math.acos(d)
        degrees_angle = rad_angle * (180.0 / math.pi)

    except ValueError:
        degrees_angle = 100

    return degrees_angle


def magnitude(vector):
    return np.linalg.norm(vector)


def check_angle(v, v_to_compare):
    is_similar = False
    angle = compare_vectors(v, v_to_compare)
    if angle <= 10:
        is_similar = True
    return is_similar


def classify_image_by_angles():
    classified_image = np.zeros((ROWS, COLS))
    locations = []
    for x in range(COLS):
        for y in range(ROWS):
            pixel = get_pixel(x, y)
            pixel = normalize_the_data(pixel)
            is_similar = check_angle(pixel, V_FABRIC)
            if is_similar:
                classified_image[y][x] = 255
                locations.append((x, y))

    return classified_image, locations


def points_on_image(locations):
    if len(locations) > 0:
        rgb = np.stack([HDR.read_band(119), HDR.read_band(52),
                        HDR.read_band(26)], axis=-1)
        rgb = rgb / rgb.max() * 1.5

        figure = plt.figure()
        figure.canvas.manager.set_window_title("RGB Image")
        plt.imshow(rgb)
        plt.scatter(*zip(*locations))
        plt.show()


def cor_func(x, y):
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    std_x = np.std(x)
    std_y = np.std(y)
    n = len(x)
    # For numpy function use this line instead of the line below
    # return np.correlate(x - mean_x, y - mean_y, mode='valid')[0] / n / (std_x * std_y)
    return correlate(x - mean_x, y - mean_y, mode='valid')[0] / n / (std_x * std_y)


def compare_curves(v, v_to_compare):
    curve_1 = v_to_compare / v_to_compare.sum()

    if v.sum() == 0:
        match_score = 0
    else:
        curve_2 = v / v.sum()

        # match_score = spearmanr(curve_1, curve_2)[0]
        # match_score = pearsonr(curve_1, curve_2)[0]
        # match_score = correlate(curve_1, curve_2, mode='same') / (np.std(curve_1) * np.std(curve_2) * len(curve_1))
        # match_score = correlate(curve_1, curve_2, mode='full')
        # match_score = np.corrcoef(curve_1, curve_2)[0, 1]
        match_score = cor_func(v, v_to_compare)

    return match_score


def compare_scores(v, v_to_compare):
    is_similar = False
    score = compare_curves(v, v_to_compare)
    if score * 100 >= 85:
        is_similar = True

    return is_similar


def classify_image_by_match_score():
    classified_image = np.zeros((ROWS, COLS))
    locations = []
    for x in range(5, 638):
        for y in range(343, 1300):
            pixel = get_pixel(x, y)
            pixel = normalize_the_data(pixel)
            is_similar = compare_scores(np.array(pixel), np.array(V_FABRIC))
            if is_similar:
                classified_image[y][x] = 255
                locations.append((x, y))

    return classified_image, locations


def create_perfect_signature(list_of_signatures):
    final_array = [sum(elem) for elem in zip(*list_of_signatures)]
    final_array = [float("{:.7f}".format(x / len(list_of_signatures))) for x in final_array]

    return final_array


def plot_vectors(list_of_vectors):
    for vector in list_of_vectors:
        plt.plot(WVL, vector)
        plt.xlabel('Wavelength')
        plt.ylabel('Reflectance')

    plt.show()


def normalize_the_data(signature):
    normal = signature / np.linalg.norm(signature)

    return [float("{:.7f}".format(x)) for x in normal]


def get_perfect_signature(file_name):
    return list(normalize_the_data(np.array((create_perfect_signature(list(read_data([file_name]).values()))))))


def peaks(signature):
    return find_peaks(signature)


def compare_peaks(peak1, peak2):
    percentage = cor_func(peak1, peak2)
    print(percentage)
    if percentage * 100 > 85:
        return True
    return False
