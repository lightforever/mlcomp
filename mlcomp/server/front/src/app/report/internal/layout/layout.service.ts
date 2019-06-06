import {EventEmitter, Injectable} from '@angular/core';

@Injectable({
  providedIn: 'root'
})
export class LayoutService{
   data_updated: EventEmitter<any> = new EventEmitter();

}
