import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def calculate_mutualinformation(x, y, bins):
    pxy, _, _ = np.histogram2d(x, y, bins)
    px, _, = np.histogram(x, bins)
    py, _, = np.histogram(y, bins)

    pxy = pxy/np.sum(pxy)
    px = px/np.sum(px)
    py = py/np.sum(py)

    pxy = pxy[np.nonzero(pxy)]
    px = px[np.nonzero(px)]
    py = py[np.nonzero(py)]

    hxy = -np.sum(pxy*np.log2(pxy))
    hx = -np.sum(px*np.log2(px))
    hy = -np.sum(py*np.log2(py))

    MI = hx+hy-hxy

    return MI

def calculate_esn_mi_input_scaling(input_data, output_data):
    if (len(input_data) != len(output_data)):
        raise ValueError("input_data and output_data do not have the same length -  {0} vs. {1}".format(len(input_data), len(output_data)))

    #Scott's rule to calculate nbins
    std_output = np.std(output_data)
    nbins = int(np.ceil(2.0/(3.5*std_output/np.power(len(input_data), 1.0/3.0))))

    mi = np.zeros(input_data.shape[1])
    for i in range(len(mi)):
         mi[i] = calculate_mutualinformation(input_data[:, i], output_data, nbins)
    scaling = mi / np.max(mi)

    return scaling

def create_2d_delay_coordinates(data, delay_dimension, tau):
    result = np.repeat(data[:, :, :, np.newaxis], repeats=delay_dimension, axis=3)

    for n in range(1, delay_dimension):
        result[:, :, :, n] = np.roll(result[:, :, :, n], n*tau, axis=0)
    result[0:delay_dimension-1,:,:] = 0

    return result

def create_1d_delay_coordinates(data, delay_dimension, tau):
    result = np.repeat(data[:, :, np.newaxis], repeats=delay_dimension, axis=2)

    for n in range(1, delay_dimension):
        result[:, :, n] = np.roll(result[:, :, n], n*tau, axis=0)
    result[0:delay_dimension-1,:,:] = 0

    return result

def create_0d_delay_coordinates(data, delay_dimension, tau):
    result = np.repeat(data[:, np.newaxis], repeats=delay_dimension, axis=1)

    for n in range(1, delay_dimension):
        result[:, n] = np.roll(result[:, n], n*tau, axis=0)
    result[0:delay_dimension-1,:] = 0

    return result

def create_rectangle_indices(range_x, range_y):
    ind_x = np.tile(range(range_x[0], range_x[1]), range_y[1] - range_y[0])
    ind_y = np.repeat(range(range_y[0], range_y[1]), range_x[1] - range_x[0])

    return ind_y, ind_x

def create_patch_indices(outer_range_x, outer_range_y, inner_range_x, inner_range_y):
    outer_ind_x = np.tile(range(outer_range_x[0], outer_range_x[1]), outer_range_y[1]-outer_range_y[0])
    outer_ind_y = np.repeat(range(outer_range_y[0], outer_range_y[1]), outer_range_x[1]-outer_range_x[0])

    inner_ind_x = np.tile(range(inner_range_x[0], inner_range_x[1]), inner_range_y[1] - inner_range_y[0])
    inner_ind_y = np.repeat(range(inner_range_y[0], inner_range_y[1]), inner_range_x[1] - inner_range_x[0])

    outer_list = [c for c in zip(outer_ind_y, outer_ind_x)]
    inner_list = [c for c in zip(inner_ind_y, inner_ind_x)]

    real_list = np.array([x for x in outer_list if x not in inner_list])
    inner_list = np.array(inner_list)

    return real_list[:,0], real_list[:,1], inner_list[:, 0], inner_list[:, 1]

def show_results(packedData, forced_clim=None):
    shape = None
    data = []

    if (type(packedData) is dict):
        for key, value in packedData.items():
            tmpItem = [key,value]
            if (type(value) is not np.ndarray):
                raise ValueError("Item for key '{0}' is not of the type numpy.ndarray".format(key))
            if (shape == None):
                shape = value.shape
            else:
                if (shape != value.shape):
                    raise ValueError("Item for key '{0}' has the shape {1} and not {2}".format(key, value.shape, shape))
            data.append(tmpItem)
    else:
        data = packedData

        for i in range(len(data)):
            if (type(data[i][1]) is not np.ndarray):
                    raise ValueError("Item for key '{0}' is not of the type numpy.ndarray".format(data[i][0]))

        shape = data[0][1].shape

    i = 0
    pause = False
    image_mode = 0

    def update_new(nextFrame):
        nonlocal i

        mat.set_data(data[image_mode][1][i])

        if (forced_clim is None):
            if (i < shape[0]-50 and i > 50):
                clb.set_clim(vmin=0, vmax=np.max(data[image_mode][1][i-50:i+50]))
        else:
            clb.set_clim(vmin = forced_clim[0], vmax=forced_clim[1])
        clb.draw_all()

        if (not pause):
            i = (i+1) % shape[0]
            sposition.set_val(i)
        return [mat]

    fig = plt.figure("main")
    ax = fig.add_subplot(111)
    mat = plt.imshow(data[0][1][0], origin="lower", interpolation="none")
    clb = plt.colorbar(mat)
    clb.set_clim(vmin=0, vmax=1)
    clb.draw_all()

    from matplotlib.widgets import Button
    from matplotlib.widgets import Slider
    class UICallback(object):
        def position_changed(self, value):
            nonlocal i
            value = int(value)
            i = value % shape[0]

        def playpause(self, event):
            nonlocal pause, bplaypause
            pause = not pause
            bplaypause.label.set_text("Play" if pause else "Pause")

        def switchsource(self, event):
            nonlocal image_mode, bswitchsource
            if (event.button == 1):
                image_mode = (image_mode + 1) % len(data)
            else:
                image_mode = (image_mode - 1) % len(data)

            bswitchsource.label.set_text(data[image_mode][0])

        def save_frame(self, event):
            nonlocal pause
            oldPause = pause
            pause = True

            from tkinter import Tk

            Tk().withdraw()
            import tkinter.filedialog as tkFileDialog
            path = tkFileDialog.asksaveasfilename(defaultextension=".pdf")
            if path is None: # asksaveasfile return `None` if dialog closed with "cancel".
                return

            savefig = plt.figure("save")
            ax = savefig.add_subplot(111)
            savemat = ax.imshow(data[image_mode][1][i], origin="lower", interpolation="none")
            saveclb = plt.colorbar(savemat)
            saveclb.set_clim(vmin=0, vmax=1)
            saveclb.draw_all()
            savefig.savefig(path, bbox_inches='tight')
            saveclb.remove()
            savefig.gca().cla()

            plt.figure("main")

            pause = oldPause

    callback = UICallback()
    axplaypause = plt.axes([0.145, 0.91, 0.10, 0.05])
    axswitchsource = plt.axes([0.645, 0.91, 0.10, 0.05])
    axsaveframe = plt.axes([0.75, 0.91, 0.15, 0.05])
    axposition = plt.axes([0.275, 0.91, 0.30, 0.05])

    bplaypause = Button(axplaypause, "Pause")
    bplaypause.on_clicked(callback.playpause)

    bswitchsource = Button(axswitchsource, data[0][0])
    bswitchsource.on_clicked(callback.switchsource)

    bsaveframe = Button(axsaveframe, "Save frame")
    bsaveframe.on_clicked(callback.save_frame)

    sposition = Slider(axposition, 'n', 0, shape[0], valinit=0, valfmt='%1.0f')
    sposition.on_changed(callback.position_changed)


    ani = animation.FuncAnimation(fig, update_new, interval=1, save_count=50)

    plt.show()

def show_results_splitscreen(packedData, forced_clim=None, name=None):
    minLength = np.inf
    data = []

    if (type(packedData) is dict):
        for key, value in packedData.items():
            tmpItem = [key,value]
            if (type(value) is not np.ndarray):
                raise ValueError("Item for key '{0}' is not of the type numpy.ndarray".format(key))
            if (shape == None):
                shape = value.shape
            else:
                if (shape != value.shape):
                    raise ValueError("Item for key '{0}' has the shape {1} and not {2}".format(key, value.shape, shape))
            data.append(tmpItem)
            minLength = min(minLength, len(tmpItem[1]))
    else:
        data = packedData

        for i in range(len(data)):
            if (type(data[i][1]) is not np.ndarray):
                raise ValueError("Item for key '{0}' is not of the type numpy.ndarray".format(data[i][0]))
            minLength = min(minLength, len(data[i][1]))

    if (len(packedData) < 2):
        print("Less than two fields submitted - switching to normal mode.")
        show_results(data, forced_clim)

    i = 0
    pause = False
    image_mode = [0, 1]

    def update_new(nextFrame):
        nonlocal i

        for n in range(2):
            matarr[n].set_data(data[image_mode[n]][1][i])

            if (forced_clim is None):
                if (i < minLength-50 and i > 50):
                    clbarr[n].set_clim(vmin=min(0, np.min(data[image_mode[n]][1][i-50:i+50]), vmax=np.max(data[image_mode[n]][1][i-50:i+50]))
            else:
                clbarr[n].set_clim(vmin = forced_clim[0], vmax=forced_clim[1])
            clbarr[n].draw_all()


        if (not pause):
            i = (i+1) % minLength
            sposition.set_val(i)
        return None

    matarr = []
    clbarr = []
    fig, axarr = plt.subplots(1,2)

    if (name == None):
        fig.canvas.set_window_title('Results')
    else:
        fig.canvas.set_window_title('Results ({0})'.format(name))

    plt.tight_layout()
    fig.subplots_adjust(top=0.85)
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    for n in range(2):
        mat = axarr[n].imshow(data[n][1][0], origin="lower", interpolation="none")
        matarr.append(mat)
        cax = make_axes_locatable(axarr[n]).append_axes("bottom", size="07.5%", pad=0.25)
        clb = plt.colorbar(mat, cax=cax, orientation="horizontal")
        cax.xaxis.set_ticks_position('bottom')
        clb.set_clim(vmin=0, vmax=1)
        clb.draw_all()
        clbarr.append(clb)
    axarr[0].set_title(data[0][0])
    axarr[1].set_title(data[1][0])

    from matplotlib.widgets import Button
    from matplotlib.widgets import Slider
    class UICallback(object):
        def position_changed(self, value):
            nonlocal i
            value = int(value)
            i = value % minLength

        def playpause(self, event):
            nonlocal pause, bplaypause
            pause = not pause
            bplaypause.label.set_text("Play" if pause else "Pause")

        def switchsource(self, event):
            nonlocal image_mode, bswitchsource
            if (event.button == 1):
                image_mode[1] = image_mode[0]
                image_mode[0] = (image_mode[0] + 1) % len(data)
            else:
                image_mode[0] = image_mode[1]
                image_mode[1] = (image_mode[1] - 1) % len(data)

            axarr[0].set_title(data[image_mode[0]][0])
            axarr[1].set_title(data[image_mode[1]][0])

        def save_frame(self, event, image_index):
            nonlocal pause
            oldPause = pause
            pause = True

            from tkinter import Tk

            Tk().withdraw()
            import tkinter.filedialog as tkFileDialog
            path = tkFileDialog.asksaveasfilename(defaultextension=".pdf")
            if path is None: # asksaveasfile return `None` if dialog closed with "cancel".
                return

            savefig = plt.figure("save")
            ax = savefig.add_subplot(111)
            savemat = ax.imshow(data[image_mode[image_index]][1][i], origin="lower", interpolation="none")
            saveclb = plt.colorbar(savemat)
            saveclb.set_clim(vmin=0, vmax=1)
            saveclb.draw_all()
            savefig.savefig(path, bbox_inches='tight')
            saveclb.remove()
            savefig.gca().cla()

            plt.figure("main")

            pause = oldPause

    callback = UICallback()
    axsaveframeleft = plt.axes([0.074, 0.895, 0.10, 0.075])
    axplaypause = plt.axes([0.180, 0.91, 0.10, 0.05])
    axposition = plt.axes([0.31, 0.91, 0.375, 0.05])
    axswitchsource = plt.axes([0.75, 0.91, 0.10, 0.05])
    axsaveframeright = plt.axes([0.86, 0.895, 0.10, 0.075])

    bplaypause = Button(axplaypause, "Pause")
    bplaypause.on_clicked(callback.playpause)

    bswitchsource = Button(axswitchsource, "switch")
    bswitchsource.on_clicked(callback.switchsource)

    bsaveframeleft = Button(axsaveframeleft, "Save\nframe")
    bsaveframeleft.on_clicked(lambda event: callback.save_frame(event, 0))

    bsaveframeright = Button(axsaveframeright, "Save\nframe")
    bsaveframeright.on_clicked(lambda event: callback.save_frame(event, 1))

    sposition = Slider(axposition, 'n', 0, minLength, valinit=0, valfmt='%1.0f')
    sposition.on_changed(callback.position_changed)

    ani = animation.FuncAnimation(fig, update_new, interval=1, save_count=50)

    plt.show()
