import {NgModule} from '@angular/core';
import {SharedModule} from "./shared.module";

import {AppRoutingModule} from './app-routing.module';

import {AppComponent} from './app.component';
import {ProjectComponent} from './project/project.component';
import {ComputerComponent} from './computer/computer.component';
import {MessagesComponent} from './messages/messages.component';
import {BrowserAnimationsModule} from '@angular/platform-browser/animations';

@NgModule({
    imports: [
        AppRoutingModule,
        SharedModule,
        BrowserAnimationsModule
    ],
    declarations: [
        AppComponent,
        ProjectComponent,
        MessagesComponent,
        ComputerComponent
    ],
    bootstrap: [AppComponent]
})
export class AppModule {
}
