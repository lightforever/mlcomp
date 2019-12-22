### Production

Mlcomp has the compiled angular files.

They are integrated into the flask-server. 

You do not have to do anything with these sources.

### Installation

If you want to change the UI, your steps:

-   Download Node.js server: 
    
    ```
    mkdir ~/programs -p && cd programs
    
    wget https://nodejs.org/dist/v10.16.0/node-v10.16.0-linux-x64.tar.xz
    ```

-   Unpack it:
       
    ```
    tar xf node-v10.16.0-linux-x64.tar.xz 
    ```
    
-   Add to $PATH:
       
    ```
    echo 'export PATH="$PATH:~/programs/node-v10.16.0-linux-x64/bin/"' >> ~/.bashrc
    source ~/.bashrc
    
    sudo ln node-v10.16.0-linux-x64/bin/node /usr/bin/node
    
    node --version
    npm --version
    ```   
-   Install npm packages:

    ```
    # Assumes mlcomp is the folder inside the project
    cd mlcomp/server/front
    npm install
    ```

    
### Start development server

Run `npm run ng serve` for a dev server. Navigate to `http://localhost:4200/`. 

The app will automatically reload if you change any of the source files.


### Build

If you are ready to replace mlcomp UI, 

Run `npm run ng -- build --prod` to build the project. 

The build artifacts will be stored in the `dist/` directory.
