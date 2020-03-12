import {NgModule} from '@angular/core';
import {SharedModule} from "../shared.module";
import {SkynetRoutingModule} from "./skynet-routing.module";
import {SkynetComponent} from "./skynet/skynet.component";
import {SpaceComponent} from "./space/space.component";
import {MemoryComponent} from "./memory/memory.component";
import {MemoryAddDialogComponent} from "./memory/memory-add-dialog";
import {SpaceAddDialogComponent} from "./space/space-add-dialog";
import {SpaceRunDialogComponent} from "./space/space-run-dialog";


@NgModule({
    imports: [
        SkynetRoutingModule,
        SharedModule
    ],
    declarations: [
        SkynetComponent,
        SpaceComponent,
        MemoryComponent,
        MemoryAddDialogComponent,
        SpaceAddDialogComponent,
        SpaceRunDialogComponent
    ],
    entryComponents: [
        MemoryAddDialogComponent,
        SpaceAddDialogComponent,
        SpaceRunDialogComponent
    ]
})
export class SkynetModule {
}