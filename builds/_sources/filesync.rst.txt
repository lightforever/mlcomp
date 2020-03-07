FileSync
======================================

.. toctree::
   :maxdepth: 2


1. Basic concepts

    File synchronization works via rsync.

    Rsync commands are being running to sync every computer.

    Install rsync on each work computer.

    In your ~/mlcomp/config/.env file there are the settings:

    - IP
    - PORT

    They must be unique for each computer.

    They must be able to connect each other using these settings.

    ::

        sudo apt-get install rsync
        # Ensure that every computer is available by SSH protocol with IP/PORT you specified
        # in the [Environment variables](#environment-variables) file.
        # rsync will perform the following commands:

        # to upload:
         rsync -vhru -e "ssh -p {target.port} -o StrictHostKeyChecking=no" \
         {folder}/ {target.user}@{target.ip}:{folder}/ --perms  --chmod=777

        # to download:
         rsync -vhru -e "ssh -p {source.port} -o StrictHostKeyChecking=no" \
         {source.user}@{source.ip}:{folder}/ {folder}/ --perms  --chmod=777

        # Please check first, the commands above work. (change parameters to yours).

2. Project settings

    Ok, you are able to sync your computers manually.

    Let's do that automatically!

    When you add/edit your project, you can set:

    -  'Sync folders'. These are the folders for synchronization.
        The paths are relative to ROOT_FOLDER (~/mlcomp). You can omit your project name. For example if your project name is `example`, you can write `data` or `data/example`. By default `data` and `models` folders are being sync.
        An example of Sync folders:

    ::

        [
             data/folder1,
             data/folder2,
             models
         ]

    -  'Ignore folders'. If you desire to ignore some folders or files,
        you can specify them here. The grammar is the same as for Sync folders.
        Example:

    ::

        [
            data/folder1/tmp
        ]

    .. image:: https://github.com/catalyst-team/mlcomp/blob/master/docs/imgs/project_settings.png?raw=true

3. Automatic synchronization

    After you performed steps 1-2, that should work in the following way:

    - each task is being executed on a worker-computer
    - if the task is completed successfully, this worker-computer will perform rsync commands to send that's content to others. You can observe these commands in Log panel.

    .. image:: https://github.com/catalyst-team/mlcomp/blob/master/docs/imgs/logs_rsync.png?raw=true

4. Manual sync via Computers tab

    You can sync manually via Computers tab.

    Sync button on the top of the window syncs all the computers.

    Moreover, each computer has its own Sync button. The computer will download all the files from other computers.

    .. image:: https://github.com/catalyst-team/mlcomp/blob/master/docs/imgs/computers_rsync.png?raw=true

5. Manual sync via Console command

    .. image:: https://github.com/catalyst-team/mlcomp/blob/master/docs/imgs/console_rsync.png?raw=true